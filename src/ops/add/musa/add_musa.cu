#include "add_musa.cuh"

#include "../../../utils.hpp"
#include "../../../utils/musa_utils.hpp"

#include <musa_runtime.h>
#include <musa_fp16.h>

#ifdef __MUSACC__
#include <musa_bf16.h>
#endif

// ---------------------------------------------------------------------------
// MUSA kernel: element-wise addition (grid-stride loop)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numel;
         i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------

namespace llaisys::ops::musa {

void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel) {
    constexpr int BLOCK = 256;
    constexpr int GRID  = 128; // grid-stride loop covers arbitrary numel

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<float><<<GRID, BLOCK>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            numel);
        break;

    case LLAISYS_DTYPE_F16:
        add_kernel<__half><<<GRID, BLOCK>>>(
            reinterpret_cast<__half *>(c),
            reinterpret_cast<const __half *>(a),
            reinterpret_cast<const __half *>(b),
            numel);
        break;

    case LLAISYS_DTYPE_BF16:
#ifdef __MUSACC__
        add_kernel<__mt_bfloat16><<<GRID, BLOCK>>>(
            reinterpret_cast<__mt_bfloat16 *>(c),
            reinterpret_cast<const __mt_bfloat16 *>(a),
            reinterpret_cast<const __mt_bfloat16 *>(b),
            numel);
#else
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
#endif
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
        break;
    }

    MUSA_CHECK(musaGetLastError());
}

} // namespace llaisys::ops::musa
