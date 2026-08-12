#include "swiglu_musa.cuh"

#include "../../../utils.hpp"
#include "../../../utils/musa_utils.hpp"

#include <musa_runtime.h>
#include <musa_fp16.h>

#ifdef __MUSACC__
#include <musa_bf16.h>
#endif

// ---------------------------------------------------------------------------
// MUSA kernel: SwiGLU  —  out = up * gate * sigmoid(gate)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numel;
         i += blockDim.x * gridDim.x) {
        float g = static_cast<float>(gate[i]);
        float u = static_cast<float>(up[i]);
        out[i] = static_cast<T>(u * g / (1.0f + expf(-g)));
    }
}

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------

namespace llaisys::ops::musa {

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel) {
    constexpr int BLOCK = 256;
    constexpr int GRID  = 128;

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<float><<<GRID, BLOCK>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up),
            numel);
        break;

    case LLAISYS_DTYPE_F16:
        swiglu_kernel<__half><<<GRID, BLOCK>>>(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(gate),
            reinterpret_cast<const __half *>(up),
            numel);
        break;

    case LLAISYS_DTYPE_BF16:
#ifdef __MUSACC__
        swiglu_kernel<__mt_bfloat16><<<GRID, BLOCK>>>(
            reinterpret_cast<__mt_bfloat16 *>(out),
            reinterpret_cast<const __mt_bfloat16 *>(gate),
            reinterpret_cast<const __mt_bfloat16 *>(up),
            numel);
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
