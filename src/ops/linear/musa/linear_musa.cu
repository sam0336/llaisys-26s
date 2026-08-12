#include "linear_musa.cuh"

#include "../../../utils.hpp"
#include "../../../utils/musa_utils.hpp"

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <mublas.h>
#include <sstream>
#include <stdexcept>

#ifdef __MUSACC__
#include <musa_bf16.h>
#endif

// ---------------------------------------------------------------------------
// Per-thread muBLAS handle (lazily created, destroyed at thread exit)
// ---------------------------------------------------------------------------

namespace {
mublasHandle_t get_mublas_handle() {
    thread_local struct HandleGuard {
        mublasHandle_t handle = nullptr;
        HandleGuard() { mublasCreate(&handle); }
        ~HandleGuard() { if (handle) mublasDestroy(handle); }
    } guard;
    return guard.handle;
}

#define MUBLAS_CHECK(call)                                                              \
    do {                                                                                \
        mublasStatus_t _s = (call);                                                     \
        if (_s != MUBLAS_STATUS_SUCCESS) {                                              \
            std::ostringstream _oss;                                                    \
            _oss << "[ERROR] muBLAS call failed at " << __FILE__ << ":" << __LINE__     \
                 << " - status " << static_cast<int>(_s);                               \
            throw std::runtime_error(_oss.str());                                       \
        }                                                                               \
    } while (0)
} // anonymous namespace

// ---------------------------------------------------------------------------
// Bias-addition kernel  (row-major, no division / modulo)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t m, size_t n) {
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < m;
         row += blockDim.x * gridDim.x) {
        T       *dst = out + row * n;
        for (size_t col = 0; col < n; ++col) {
            dst[col] = static_cast<T>(static_cast<float>(dst[col])
                                    + static_cast<float>(bias[col]));
        }
    }
}

// ---------------------------------------------------------------------------
// MUSA kernel: naive linear fallback (row-major, no division / modulo)
//               Used when muBLAS does not support a particular dtype.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void linear_kernel(T *out, const T *in, const T *weight,
                               const T *bias, size_t m, size_t n, size_t k) {
    size_t total = m * n;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        size_t i = idx / n;
        size_t j = idx % n;
        const T *src = in + i * k;
        float sum = (bias ? static_cast<float>(bias[j]) : 0.0f);
        for (size_t kk = 0; kk < k; ++kk) {
            sum += static_cast<float>(src[kk])
                 * static_cast<float>(weight[j * k + kk]);
        }
        out[idx] = static_cast<T>(sum);
    }
}

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------

namespace llaisys::ops::musa {

void linear(std::byte *out, const std::byte *in,
            const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t m, size_t n, size_t k) {
    if (m == 0 || n == 0 || k == 0) return;

    constexpr int BLOCK = 256;
    constexpr int GRID  = 128;

    switch (dtype) {
    // ---- F32: muBLAS SGEMM + bias kernel --------------------------------
    case LLAISYS_DTYPE_F32: {
        const float alpha = 1.0f, beta = 0.0f;
        mublasHandle_t handle = get_mublas_handle();

        // Y[m×n] = X[m×k] × W[n×k]^T
        // muBLAS column-major convention: C^T = B^T × A^T → C = A × B
        MUBLAS_CHECK(mublasSgemm(handle,
            MUBLAS_OP_T, MUBLAS_OP_N,
            static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
            &alpha,
            reinterpret_cast<const float *>(weight), static_cast<int>(k),
            reinterpret_cast<const float *>(in),     static_cast<int>(k),
            &beta,
            reinterpret_cast<float *>(out),          static_cast<int>(n)));

        if (bias) {
            add_bias_kernel<float><<<GRID, BLOCK>>>(
                reinterpret_cast<float *>(out),
                reinterpret_cast<const float *>(bias),
                m, n);
        }
        break;
    }

    // ---- F16: naive kernel (muBLAS GemmEx F16 returns status 2 on MUSA) ----
    case LLAISYS_DTYPE_F16: {
        linear_kernel<__half><<<GRID, BLOCK>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(in),
            reinterpret_cast<const __half *>(weight),
            reinterpret_cast<const __half *>(bias),
            m, n, k);
        break;
    }

    // ---- BF16: muBLAS on supported hardware, fallback to naive kernel ----
    case LLAISYS_DTYPE_BF16:
#ifdef __MUSACC__
    {
        // Try muBLAS first.
        int device;
        musaGetDevice(&device);
        musaDeviceProp prop;
        musaGetDeviceProperties(&prop, device);
        if (prop.major >= 2) {
            const float alpha = 1.0f, beta = 0.0f;
            mublasHandle_t handle = get_mublas_handle();
            mublasStatus_t stat = mublasGemmEx(handle,
                MUBLAS_OP_T, MUBLAS_OP_N,
                static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                &alpha,
                reinterpret_cast<const void *>(weight), MUSA_R_16BF, static_cast<int>(k),
                reinterpret_cast<const void *>(in),     MUSA_R_16BF, static_cast<int>(k),
                &beta,
                reinterpret_cast<void *>(out),          MUSA_R_16BF, static_cast<int>(n),
                MUSA_R_32F, MUBLAS_GEMM_DEFAULT);
            if (stat == MUBLAS_STATUS_SUCCESS) {
                if (bias) {
                    add_bias_kernel<__mt_bfloat16><<<GRID, BLOCK>>>(
                        reinterpret_cast<__mt_bfloat16 *>(out),
                        reinterpret_cast<const __mt_bfloat16 *>(bias),
                        m, n);
                }
                break;
            }
            // Fall through to naive kernel on muBLAS failure.
        }
        linear_kernel<__mt_bfloat16><<<GRID, BLOCK>>>(
            reinterpret_cast<__mt_bfloat16 *>(out),
            reinterpret_cast<const __mt_bfloat16 *>(in),
            reinterpret_cast<const __mt_bfloat16 *>(weight),
            reinterpret_cast<const __mt_bfloat16 *>(bias),
            m, n, k);
    }
#else
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
#endif
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        break;
    }

    MUSA_CHECK(musaGetLastError());
}

} // namespace llaisys::ops::musa
