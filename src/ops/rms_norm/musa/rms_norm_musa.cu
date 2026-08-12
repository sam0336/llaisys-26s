#include "rms_norm_musa.cuh"

#include "../../../utils.hpp"
#include "../../../utils/musa_utils.hpp"

#include <musa_runtime.h>
#include <musa_fp16.h>

#ifdef __MUSACC__
#include <musa_bf16.h>
#endif

// ---------------------------------------------------------------------------
// MUSA kernel: RMS Normalization (one block per row, parallel reduction)
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight,
                                 float eps, size_t m, size_t d) {
    const size_t row = blockIdx.x;
    if (row >= m) return;

    const int tid = threadIdx.x;

    // ---- Step 1: compute sum of squares via parallel reduction ----
    __shared__ float s_sum_sq[BLOCK_SIZE];
    float local_sum = 0.0f;

    for (size_t j = tid; j < d; j += blockDim.x) {
        float val = static_cast<float>(in[row * d + j]);
        local_sum += val * val;
    }
    s_sum_sq[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    float rms = sqrtf(s_sum_sq[0] / static_cast<float>(d) + eps);

    // ---- Step 2: apply normalization ----
    for (size_t j = tid; j < d; j += blockDim.x) {
        float val = static_cast<float>(in[row * d + j]);
        float w   = static_cast<float>(weight[j]);
        out[row * d + j] = static_cast<T>(val * w / rms);
    }
}

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------

namespace llaisys::ops::musa {

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, llaisysDataType_t dtype, size_t m, size_t d) {
    if (m == 0 || d == 0) return;

    constexpr int BLOCK = 256;
    const int grid = static_cast<int>(m);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<float, BLOCK><<<grid, BLOCK>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            eps, m, d);
        break;

    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<__half, BLOCK><<<grid, BLOCK>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(in),
            reinterpret_cast<const __half *>(weight),
            eps, m, d);
        break;

    case LLAISYS_DTYPE_BF16:
#ifdef __MUSACC__
        rms_norm_kernel<__mt_bfloat16, BLOCK><<<grid, BLOCK>>>(
            reinterpret_cast<__mt_bfloat16 *>(out),
            reinterpret_cast<const __mt_bfloat16 *>(in),
            reinterpret_cast<const __mt_bfloat16 *>(weight),
            eps, m, d);
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
