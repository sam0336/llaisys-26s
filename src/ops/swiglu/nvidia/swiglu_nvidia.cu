#include "swiglu_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda_bf16.h>
#endif

// ---------------------------------------------------------------------------
// CUDA kernel: SwiGLU  —  out = up * gate * sigmoid(gate)
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

namespace llaisys::ops::nvidia {

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
#if __CUDACC_VER_MAJOR__ >= 11
        swiglu_kernel<__nv_bfloat16><<<GRID, BLOCK>>>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(gate),
            reinterpret_cast<const __nv_bfloat16 *>(up),
            numel);
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
