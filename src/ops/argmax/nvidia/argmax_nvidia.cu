#include "argmax_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>

#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda_bf16.h>
#endif

// ---------------------------------------------------------------------------
// CUDA kernel: multi-block partial argmax reduction
// ---------------------------------------------------------------------------

/**
 * @brief Phase 1 — each block reduces a contiguous chunk of the input
 *        and writes its local (idx, val) to the partial-results arrays.
 *
 * @tparam T          Element type (float, __half, __nv_bfloat16).
 * @tparam BLOCK_SIZE Threads per block (shared-memory arrays are sized to match).
 */
template <typename T, int BLOCK_SIZE>
__global__ void argmax_partial_kernel(
    const T *vals, size_t numel, size_t chunk_size,
    int64_t *partial_idxs, T *partial_vals)
{
    __shared__ T       s_vals[BLOCK_SIZE];
    __shared__ int64_t s_idxs[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const size_t chunk_start = blockIdx.x * chunk_size;
    const size_t chunk_end   = min(chunk_start + chunk_size, numel);

    // Every thread starts with the chunk's first element as a floor.
    T       best_val = vals[chunk_start];
    int64_t best_idx = static_cast<int64_t>(chunk_start);

    for (size_t i = chunk_start + tid; i < chunk_end; i += BLOCK_SIZE) {
        if (static_cast<float>(vals[i]) > static_cast<float>(best_val)) {
            best_val = vals[i];
            best_idx = static_cast<int64_t>(i);
        }
    }

    // Write local best to shared memory for block-level tree reduction.
    s_vals[tid] = best_val;
    s_idxs[tid] = best_idx;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (static_cast<float>(s_vals[tid + s])
                > static_cast<float>(s_vals[tid])) {
                s_vals[tid] = s_vals[tid + s];
                s_idxs[tid] = s_idxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_idxs[blockIdx.x] = s_idxs[0];
        partial_vals[blockIdx.x] = s_vals[0];
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel: final reduction over partial results
// ---------------------------------------------------------------------------

/**
 * @brief Phase 2 — single-block reduction over the per-block partial results
 *        to produce the global argmax.
 */
template <typename T, int BLOCK_SIZE>
__global__ void argmax_final_kernel(
    const int64_t *partial_idxs, const T *partial_vals,
    int num_partials,
    int64_t *max_idx, T *max_val)
{
    __shared__ T       s_vals[BLOCK_SIZE];
    __shared__ int64_t s_idxs[BLOCK_SIZE];

    const int tid = threadIdx.x;

    T       best_val = partial_vals[0];
    int64_t best_idx = partial_idxs[0];

    for (int i = tid; i < num_partials; i += BLOCK_SIZE) {
        if (static_cast<float>(partial_vals[i]) > static_cast<float>(best_val)) {
            best_val = partial_vals[i];
            best_idx = partial_idxs[i];
        }
    }

    s_vals[tid] = best_val;
    s_idxs[tid] = best_idx;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (static_cast<float>(s_vals[tid + s])
                > static_cast<float>(s_vals[tid])) {
                s_vals[tid] = s_vals[tid + s];
                s_idxs[tid] = s_idxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_idx = s_idxs[0];
        *max_val = s_vals[0];
    }
}

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------

namespace llaisys::ops::nvidia {

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t dtype, size_t numel) {
    if (numel == 0) return;

    constexpr int BLOCK    = 256;
    constexpr int MAX_GRID = 256;

    int grid = static_cast<int>(std::min(
        (numel + BLOCK - 1) / BLOCK,
        static_cast<size_t>(MAX_GRID)));
    if (grid < 1) grid = 1;

    size_t chunk_size = (numel + grid - 1) / grid;

    switch (dtype) {
    case LLAISYS_DTYPE_F32: {
        float   *partial_vals;
        int64_t *partial_idxs;
        CUDA_CHECK(cudaMalloc(&partial_idxs, grid * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&partial_vals, grid * sizeof(float)));

        argmax_partial_kernel<float, BLOCK><<<grid, BLOCK>>>(
            reinterpret_cast<const float *>(vals),
            numel, chunk_size,
            partial_idxs, partial_vals);

        argmax_final_kernel<float, BLOCK><<<1, BLOCK>>>(
            partial_idxs, partial_vals, grid,
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<float *>(max_val));
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(partial_vals));
        CUDA_CHECK(cudaFree(partial_idxs));
        break;
    }

    case LLAISYS_DTYPE_F16: {
        __half  *partial_vals;
        int64_t *partial_idxs;
        CUDA_CHECK(cudaMalloc(&partial_idxs, grid * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&partial_vals, grid * sizeof(__half)));

        argmax_partial_kernel<__half, BLOCK><<<grid, BLOCK>>>(
            reinterpret_cast<const __half *>(vals),
            numel, chunk_size,
            partial_idxs, partial_vals);

        argmax_final_kernel<__half, BLOCK><<<1, BLOCK>>>(
            partial_idxs, partial_vals, grid,
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<__half *>(max_val));
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(partial_vals));
        CUDA_CHECK(cudaFree(partial_idxs));
        break;
    }

    case LLAISYS_DTYPE_BF16:
#if __CUDACC_VER_MAJOR__ >= 11
    {
        __nv_bfloat16 *partial_vals;
        int64_t       *partial_idxs;
        CUDA_CHECK(cudaMalloc(&partial_idxs, grid * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&partial_vals, grid * sizeof(__nv_bfloat16)));

        argmax_partial_kernel<__nv_bfloat16, BLOCK><<<grid, BLOCK>>>(
            reinterpret_cast<const __nv_bfloat16 *>(vals),
            numel, chunk_size,
            partial_idxs, partial_vals);

        argmax_final_kernel<__nv_bfloat16, BLOCK><<<1, BLOCK>>>(
            partial_idxs, partial_vals, grid,
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<__nv_bfloat16 *>(max_val));
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(partial_vals));
        CUDA_CHECK(cudaFree(partial_idxs));
    }
#else
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
#endif
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        break;
    }
}

} // namespace llaisys::ops::nvidia
