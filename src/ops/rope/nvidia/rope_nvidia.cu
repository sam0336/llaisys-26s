#include "rope_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <vector>

#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda_bf16.h>
#endif

// ---------------------------------------------------------------------------
// CUDA kernel: RoPE  (freqs precomputed by host)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                             const float *freqs, size_t seqlen, size_t nheads,
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
            float angle = pos / freqs[j];
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

    // Precompute frequency table on the host (half_d unique values).
    size_t half_d = d / 2;
    std::vector<float> freqs_host(half_d);
    for (size_t j = 0; j < half_d; ++j) {
        freqs_host[j] = std::pow(theta, 2.0f * static_cast<float>(j)
                                       / static_cast<float>(d));
    }

    // Copy frequency table to device.
    float *freqs_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&freqs_dev, half_d * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(freqs_dev, freqs_host.data(),
                          half_d * sizeof(float), cudaMemcpyHostToDevice));

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<float><<<GRID, BLOCK>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            freqs_dev, seqlen, nheads, d);
        break;

    case LLAISYS_DTYPE_F16:
        rope_kernel<__half><<<GRID, BLOCK>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            freqs_dev, seqlen, nheads, d);
        break;

    case LLAISYS_DTYPE_BF16:
#if __CUDACC_VER_MAJOR__ >= 11
        rope_kernel<__nv_bfloat16><<<GRID, BLOCK>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            freqs_dev, seqlen, nheads, d);
#else
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
#endif
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        break;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(freqs_dev));
}

} // namespace llaisys::ops::nvidia
