#include "linear_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <sstream>
#include <stdexcept>

#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda_bf16.h>
#endif

// ---------------------------------------------------------------------------
// Per-thread cuBLAS handle (lazily created, destroyed at thread exit)
// ---------------------------------------------------------------------------

namespace {
cublasHandle_t get_cublas_handle() {
    thread_local struct HandleGuard {
        cublasHandle_t handle = nullptr;
        HandleGuard() { cublasCreate(&handle); }
        ~HandleGuard() { if (handle) cublasDestroy(handle); }
    } guard;
    return guard.handle;
}

#define CUBLAS_CHECK(call)                                                              \
    do {                                                                                \
        cublasStatus_t _s = (call);                                                     \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                              \
            std::ostringstream _oss;                                                    \
            _oss << "[ERROR] cuBLAS call failed at " << __FILE__ << ":" << __LINE__     \
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
// CUDA kernel: naive linear fallback (row-major, no division / modulo)
//               Used when cuBLAS does not support a particular dtype.
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

namespace llaisys::ops::nvidia {

void linear(std::byte *out, const std::byte *in,
            const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t m, size_t n, size_t k) {
    if (m == 0 || n == 0 || k == 0) return;

    constexpr int BLOCK = 256;
    constexpr int GRID  = 128;

    switch (dtype) {
    // ---- F32: cuBLAS SGEMM + bias kernel --------------------------------
    case LLAISYS_DTYPE_F32: {
        const float alpha = 1.0f, beta = 0.0f;
        cublasHandle_t handle = get_cublas_handle();

        // Y[m×n] = X[m×k] × W[n×k]^T
        // cuBLAS column-major convention: C^T = B^T × A^T → C = A × B
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
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

    // ---- F16: cuBLAS GemmEx + bias kernel --------------------------------
    case LLAISYS_DTYPE_F16: {
        const float alpha = 1.0f, beta = 0.0f;
        cublasHandle_t handle = get_cublas_handle();

        CUBLAS_CHECK(cublasGemmEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
            &alpha,
            reinterpret_cast<const void *>(weight), CUDA_R_16F, static_cast<int>(k),
            reinterpret_cast<const void *>(in),     CUDA_R_16F, static_cast<int>(k),
            &beta,
            reinterpret_cast<void *>(out),          CUDA_R_16F, static_cast<int>(n),
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        if (bias) {
            add_bias_kernel<__half><<<GRID, BLOCK>>>(
                reinterpret_cast<__half *>(out),
                reinterpret_cast<const __half *>(bias),
                m, n);
        }
        break;
    }

    // ---- BF16: cuBLAS on Ampere+ (SM 80), fallback to naive kernel --------
    case LLAISYS_DTYPE_BF16:
#if __CUDACC_VER_MAJOR__ >= 11
    {
        // Try cuBLAS first (available on SM 80+ / Ampere+).
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        if (prop.major >= 8) {
            const float alpha = 1.0f, beta = 0.0f;
            cublasHandle_t handle = get_cublas_handle();
            // cuBLAS might not support BF16 on all Ampere+ GPUs;
            // fall back to naive kernel if it fails.
            cublasStatus_t stat = cublasGemmEx(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                &alpha,
                reinterpret_cast<const void *>(weight), CUDA_R_16BF, static_cast<int>(k),
                reinterpret_cast<const void *>(in),     CUDA_R_16BF, static_cast<int>(k),
                &beta,
                reinterpret_cast<void *>(out),          CUDA_R_16BF, static_cast<int>(n),
                CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
            if (stat == CUBLAS_STATUS_SUCCESS) {
                if (bias) {
                    add_bias_kernel<__nv_bfloat16><<<GRID, BLOCK>>>(
                        reinterpret_cast<__nv_bfloat16 *>(out),
                        reinterpret_cast<const __nv_bfloat16 *>(bias),
                        m, n);
                }
                break;
            }
            // Fall through to naive kernel on cuBLAS failure.
        }
        linear_kernel<__nv_bfloat16><<<GRID, BLOCK>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const __nv_bfloat16 *>(weight),
            reinterpret_cast<const __nv_bfloat16 *>(bias),
            m, n, k);
        break;
    }
#else
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        break;
#endif

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        break;
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
