#include "embedding_musa.cuh"

#include "../../../utils.hpp"
#include "../../../utils/musa_utils.hpp"

#include <musa_runtime.h>
#include <musa_fp16.h>

#ifdef __MUSACC__
#include <musa_bf16.h>
#endif

// ---------------------------------------------------------------------------
// MUSA kernel: embedding lookup (row-major, no division / modulo)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight,
                                  size_t numel, size_t embedding_dim) {
    for (size_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < numel;
         row += blockDim.x * gridDim.x) {
        int64_t idx = index[row];
        const T *src = weight + idx * embedding_dim;
        T       *dst = out   + row * embedding_dim;
        for (size_t col = 0; col < embedding_dim; ++col) {
            dst[col] = src[col];
        }
    }
}

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------

namespace llaisys::ops::musa {

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t numel, size_t embedding_dim) {
    constexpr int BLOCK = 256;
    constexpr int GRID  = 128;

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<float><<<GRID, BLOCK>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const float *>(weight),
            numel, embedding_dim);
        break;

    case LLAISYS_DTYPE_F16:
        embedding_kernel<__half><<<GRID, BLOCK>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const __half *>(weight),
            numel, embedding_dim);
        break;

    case LLAISYS_DTYPE_BF16:
#ifdef __MUSACC__
        embedding_kernel<__mt_bfloat16><<<GRID, BLOCK>>>(
            reinterpret_cast<__mt_bfloat16 *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const __mt_bfloat16 *>(weight),
            numel, embedding_dim);
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
