#include "self_attention_musa.cuh"

#include "../../../utils.hpp"
#include "../../../utils/musa_utils.hpp"

#include <musa_runtime.h>
#include <musa_fp16.h>

#ifdef __MUSACC__
#include <musa_bf16.h>
#endif

// ===================================================================
// Kernel 1 —  Q @ K^T * scale  (with causal mask baked in)
//             Row-major restructuring: (s,h) grid-stride → j → dim.
// ===================================================================

template <typename T>
__global__ void qk_scores_kernel(float *scores,
                                  const T *q, const T *k,
                                  float scale,
                                  size_t seqlen, size_t total_len,
                                  size_t nhead, size_t nkvhead, size_t d) {
    size_t nrep    = nhead / nkvhead;
    size_t offset  = total_len - seqlen;  // computed once, not per element
    size_t stride_h = total_len;
    size_t stride_s = nhead * total_len;

    // Grid-stride over (s, h) pairs — 1 division + 1 modulo per pair.
    for (size_t sh = blockIdx.x * blockDim.x + threadIdx.x;
         sh < seqlen * nhead;
         sh += blockDim.x * gridDim.x) {
        size_t s  = sh / nhead;
        size_t h  = sh % nhead;
        size_t kh = h / nrep;

        size_t q_base     = (s * nhead + h) * d;
        size_t score_base = s * stride_s + h * stride_h;

        for (size_t j = 0; j < total_len; ++j) {
            // Causal mask: j cannot attend beyond s + offset.
            if (j > s + offset) {
                scores[score_base + j] = -INFINITY;
                continue;
            }

            double dot = 0.0;
            for (size_t dim = 0; dim < d; ++dim) {
                dot += static_cast<double>(q[q_base + dim])
                     * static_cast<double>(k[(j * nkvhead + kh) * d + dim]);
            }
            scores[score_base + j] = static_cast<float>(dot * static_cast<double>(scale));
        }
    }
}

// ===================================================================
// Kernel 2 — causal softmax (in-place, one block per row of scores)
//            BLOCK_SIZE templatized to eliminate hardcoded shared mem.
// ===================================================================

template <int BLOCK_SIZE>
__global__ void causal_softmax_kernel(float *scores,
                                       size_t seqlen, size_t nhead,
                                       size_t total_len) {
    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
                  "BLOCK_SIZE must be a power of 2");

    size_t stride_h = total_len;
    size_t stride_s = nhead * total_len;
    size_t total_rows = seqlen * nhead;
    size_t row = blockIdx.x;

    if (row >= total_rows) return;

    size_t s    = row / nhead;
    size_t h    = row % nhead;
    size_t base = s * stride_s + h * stride_h;

    int tid = threadIdx.x;

    // ---- Pass 1: find max (double reduction for precision) ----
    float max_val = -INFINITY;
    for (size_t j = tid; j < total_len; j += blockDim.x) {
        float val = scores[base + j];
        if (val > max_val) max_val = val;
    }

    __shared__ double s_max[BLOCK_SIZE];
    s_max[tid] = static_cast<double>(max_val);
    __syncthreads();
    for (int ss = blockDim.x / 2; ss > 0; ss >>= 1) {
        if (tid < ss && s_max[tid + ss] > s_max[tid])
            s_max[tid] = s_max[tid + ss];
        __syncthreads();
    }
    max_val = static_cast<float>(s_max[0]);
    __syncthreads();

    // ---- Pass 2: exp + sum ----
    double local_sum = 0.0;
    for (size_t j = tid; j < total_len; j += blockDim.x) {
        float val = scores[base + j];
        float exp_val = (val == -INFINITY) ? 0.0f : expf(val - max_val);
        scores[base + j] = exp_val;
        local_sum += static_cast<double>(exp_val);
    }

    __shared__ double s_sum[BLOCK_SIZE];
    s_sum[tid] = local_sum;
    __syncthreads();
    for (int ss = blockDim.x / 2; ss > 0; ss >>= 1) {
        if (tid < ss) s_sum[tid] += s_sum[tid + ss];
        __syncthreads();
    }
    double total_sum = s_sum[0];
    __syncthreads();

    // ---- Pass 3: normalize (use double division for precision) ----
    for (size_t j = tid; j < total_len; j += blockDim.x) {
        if (total_sum > 0.0)
            scores[base + j] = static_cast<float>(
                static_cast<double>(scores[base + j]) / total_sum);
        else
            scores[base + j] = 0.0f;  // all-masked row -> zero
    }
}

// ===================================================================
// Kernel 3 — attn_val = softmax(scores) @ V
//             (s,h) grid-stride → dim → j.
// ===================================================================

template <typename T>
__global__ void combine_kernel(T *attn_val, const float *scores,
                                const T *v,
                                size_t seqlen, size_t total_len,
                                size_t nhead, size_t nkvhead,
                                size_t dv) {
    size_t nrep    = nhead / nkvhead;
    size_t stride_h = total_len;
    size_t stride_s = nhead * total_len;

    // Grid-stride over (s, h) pairs.
    for (size_t sh = blockIdx.x * blockDim.x + threadIdx.x;
         sh < seqlen * nhead;
         sh += blockDim.x * gridDim.x) {
        size_t s  = sh / nhead;
        size_t h  = sh % nhead;
        size_t kh = h / nrep;

        size_t score_base = s * stride_s + h * stride_h;
        size_t out_base   = (s * nhead + h) * dv;

        for (size_t dim = 0; dim < dv; ++dim) {
            double val = 0.0;
            for (size_t j = 0; j < total_len; ++j) {
                float sc = scores[score_base + j];
                if (sc > 0.0f) {
                    val += static_cast<double>(sc) * static_cast<double>(
                        v[(j * nkvhead + kh) * dv + dim]);
                }
            }
            attn_val[out_base + dim] = static_cast<T>(static_cast<float>(val));
        }
    }
}

// ===================================================================
// Host dispatch
// ===================================================================

namespace llaisys::ops::musa {

void self_attention(std::byte *attn_val, const std::byte *q,
                    const std::byte *k, const std::byte *v,
                    float scale, llaisysDataType_t dtype,
                    size_t seqlen, size_t total_len, size_t nhead,
                    size_t nkvhead, size_t d, size_t dv) {

    // ---- Input validation ----
    if (seqlen == 0 || nhead == 0 || nkvhead == 0 ||
        d == 0 || dv == 0 || total_len == 0)
        return;
    if (total_len < seqlen) return;
    if (nhead % nkvhead != 0) return;

    // Allocate temporary device memory for the scores matrix.
    size_t scores_elems = seqlen * nhead * total_len;
    size_t scores_bytes = scores_elems * sizeof(float);
    float *scores_dev = nullptr;
    MUSA_CHECK(musaMalloc(&scores_dev, scores_bytes));

    constexpr int BLOCK    = 256;
    constexpr int GRID     = 128;
    constexpr int SM_BLOCK = 256;  // for softmax (one block per row)

    // ---- Stage 1: QK^T scores ----
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        qk_scores_kernel<float><<<GRID, BLOCK>>>(
            scores_dev,
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            scale, seqlen, total_len, nhead, nkvhead, d);
        break;
    case LLAISYS_DTYPE_F16:
        qk_scores_kernel<__half><<<GRID, BLOCK>>>(
            scores_dev,
            reinterpret_cast<const __half *>(q),
            reinterpret_cast<const __half *>(k),
            scale, seqlen, total_len, nhead, nkvhead, d);
        break;
    case LLAISYS_DTYPE_BF16:
#ifdef __MUSACC__
        qk_scores_kernel<__mt_bfloat16><<<GRID, BLOCK>>>(
            scores_dev,
            reinterpret_cast<const __mt_bfloat16 *>(q),
            reinterpret_cast<const __mt_bfloat16 *>(k),
            scale, seqlen, total_len, nhead, nkvhead, d);
        break;
#else
        MUSA_CHECK(musaFree(scores_dev));
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        break;
#endif
    default:
        MUSA_CHECK(musaFree(scores_dev));
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        break;
    }
    MUSA_CHECK(musaGetLastError());

    // ---- Stage 2: causal softmax (one block per row) ----
    {
        size_t rows = seqlen * nhead;
        causal_softmax_kernel<SM_BLOCK><<<rows, SM_BLOCK>>>(
            scores_dev, seqlen, nhead, total_len);
        MUSA_CHECK(musaGetLastError());
    }

    // ---- Stage 3: combine with V ----
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        combine_kernel<float><<<GRID, BLOCK>>>(
            reinterpret_cast<float *>(attn_val), scores_dev,
            reinterpret_cast<const float *>(v),
            seqlen, total_len, nhead, nkvhead, dv);
        break;
    case LLAISYS_DTYPE_F16:
        combine_kernel<__half><<<GRID, BLOCK>>>(
            reinterpret_cast<__half *>(attn_val), scores_dev,
            reinterpret_cast<const __half *>(v),
            seqlen, total_len, nhead, nkvhead, dv);
        break;
    case LLAISYS_DTYPE_BF16:
#ifdef __MUSACC__
        combine_kernel<__mt_bfloat16><<<GRID, BLOCK>>>(
            reinterpret_cast<__mt_bfloat16 *>(attn_val), scores_dev,
            reinterpret_cast<const __mt_bfloat16 *>(v),
            seqlen, total_len, nhead, nkvhead, dv);
        break;
#else
        MUSA_CHECK(musaFree(scores_dev));
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        break;
#endif
    default:
        MUSA_CHECK(musaFree(scores_dev));
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
        break;
    }
    MUSA_CHECK(musaGetLastError());

    // Free temporary scores buffer.
    MUSA_CHECK(musaFree(scores_dev));
}

} // namespace llaisys::ops::musa
