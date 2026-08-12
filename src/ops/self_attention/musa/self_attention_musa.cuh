#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::musa {

/**
 * @brief Self-attention with causal masking on MUSA GPU.
 *
 * Computes:  scores = Q @ K^T * scale
 *            scores = causal_softmax(scores)
 *            attn_val = scores @ V
 *
 * Supports Grouped Query Attention (GQA) when nhead > nkvhead.
 *
 * @param attn_val  Output tensor (device pointer), 3-D [seqlen, nhead, dv].
 * @param q         Query tensor (device pointer), 3-D [seqlen, nhead, d].
 * @param k         Key tensor (device pointer), 3-D [total_len, nkvhead, d].
 * @param v         Value tensor (device pointer), 3-D [total_len, nkvhead, dv].
 * @param scale     Scaling factor (typically 1/sqrt(d)).
 * @param dtype     Data type.
 * @param seqlen    Number of query positions.
 * @param total_len Total key/value length (including cache).
 * @param nhead     Number of query heads.
 * @param nkvhead   Number of key/value heads.
 * @param d         Head dimension for Q and K.
 * @param dv        Head dimension for V.
 */
void self_attention(std::byte *attn_val, const std::byte *q,
                    const std::byte *k, const std::byte *v,
                    float scale, llaisysDataType_t dtype,
                    size_t seqlen, size_t total_len, size_t nhead,
                    size_t nkvhead, size_t d, size_t dv);

} // namespace llaisys::ops::musa
