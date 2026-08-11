#include "rope_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda_bf16.h>
#endif

// ---------------------------------------------------------------------------
// CUDA kernel: RoPE  (freqs precomputed by host)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                             float theta, size_t seqlen, size_t nheads,
                             size_t d) {
    size_t half_d = d / 2;

    for (size_t sh = blockIdx.x * blockDim.x + threadIdx.x;
         sh < seqlen * nheads;
         sh += blockDim.x * gridDim.x) {
        size_t s = sh / nheads;
        size_t h = sh % nheads;
        float pos = static_cast<float>(pos_ids[s]);
        size_t base = (s * nheads + h) * d;

        for (size_t j = 0; j < half_d; ++j) {
            // Compute angle directly on GPU to match PyTorch's device-side
            // precision (host std::pow ≠ device powf).
            float angle = pos / powf(theta,
                                     2.0f * static_cast<float>(j)
                                     / static_cast<float>(d));
            float c     = cosf(angle);
            float si    = sinf(angle);

            float a = static_cast<float>(in[base + j]);
            float b = static_cast<float>(in[base + half_d + j]);

            out[base + j]          = static_cast<T>(a * c - b * si);
            out[base + half_d + j] = static_cast<T>(b * c + a * si);
        }
    }
}

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------

namespace llaisys::ops::nvidia {

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          float theta, llaisysDataType_t dtype,
          size_t seqlen, size_t nheads, size_t d) {
    if (seqlen == 0 || nheads == 0 || d == 0) return;

    constexpr int BLOCK = 256;
    constexpr int GRID  = 128;

    // Angle is computed inline on GPU (pos / theta^(2j/d) via powf)
    // so it matches PyTorch's device-side precision.

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<float><<<GRID, BLOCK>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            theta, seqlen, nheads, d);
        break;

    case LLAISYS_DTYPE_F16:
        rope_kernel<__half><<<GRID, BLOCK>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            theta, seqlen, nheads, d);
        break;

    case LLAISYS_DTYPE_BF16:
#if __CUDACC_VER_MAJOR__ >= 11
        rope_kernel<__nv_bfloat16><<<GRID, BLOCK>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            theta, seqlen, nheads, d);
#else
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
#endif
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        break;
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
