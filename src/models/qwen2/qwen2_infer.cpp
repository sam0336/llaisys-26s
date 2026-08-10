#include "qwen2_infer.hpp"

#include "../../llaisys/llaisys_tensor.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <cstring>

namespace llaisys::models {

namespace {
/**
 * Unwrap a llaisysTensor_t (C handle = LlaisysTensor*) into the underlying
 * tensor_t (shared_ptr<Tensor>) that operators expect.
 *
 * Returns an empty shared_ptr when the handle is nullptr — this is used
 * for optional bias tensors.
 */
inline tensor_t unwrap(llaisysTensor_t t) {
    if (t) {
        return t->tensor;
    }
    return tensor_t();
}
} // anonymous namespace

int64_t qwen2_infer(Qwen2Model *model, const int64_t *token_ids, size_t ntoken) {
    // Convenience aliases for the meta fields and weight struct
    const auto &meta = model->_meta;
    const auto &W    = model->_weights;

    // ------------------------------------------------------------------
    // Process each token one at a time so that the pre-allocated
    // [1, ...] intermediate tensors are always the right shape.
    // ------------------------------------------------------------------
    int64_t next_token = 0;

    for (size_t step = 0; step < ntoken; step++) {
        // ---- Position of this token in the full sequence ----
        size_t   pos     = model->_cached_len + step;
        int64_t  token   = token_ids[step];
        int64_t  pos_i64 = static_cast<int64_t>(pos);

        // ---- Load token id and position id onto device ----
        model->_token_ids_tensor->load(&token);
        model->_pos_ids_tensor->load(&pos_i64);

        // ==============================================================
        // 1. Token Embedding   _hidden [1, hs] ← in_embed[token_id]
        // ==============================================================
        ops::embedding(model->_hidden,
                       model->_token_ids_tensor,
                       unwrap(W.in_embed));

        // ==============================================================
        // 2. Transformer layers
        // ==============================================================
        for (size_t l = 0; l < meta.nlayer; l++) {
            // ----------------------------------------------------------
            // 2a. RMS Normalization (input_layernorm)
            // ----------------------------------------------------------
            ops::rms_norm(model->_norm_out,
                          model->_hidden,
                          unwrap(W.attn_norm_w[l]),
                          meta.epsilon);

            // ----------------------------------------------------------
            // 2b. Q / K / V Linear projections
            //
            //     Weight layout (row-major, as stored in safetensors):
            //       attn_q_w  [nh   * dh, hs]
            //       attn_k_w  [nkvh * dh, hs]
            //       attn_v_w  [nkvh * dh, hs]
            //
            //     Pre-allocated tensors have shape [1, heads, dh].
            //     For linear() we view them flat  →  [1, heads*dh].
            //     After the call, the original 3D view is still valid.
            // ----------------------------------------------------------

            // --- Q ---
            ops::linear(model->_q_flat, model->_norm_out,
                        unwrap(W.attn_q_w[l]),
                        unwrap(W.attn_q_b[l]));

            // --- K ---
            ops::linear(model->_k_flat, model->_norm_out,
                        unwrap(W.attn_k_w[l]),
                        unwrap(W.attn_k_b[l]));

            // --- V ---
            ops::linear(model->_v_flat, model->_norm_out,
                        unwrap(W.attn_v_w[l]),
                        unwrap(W.attn_v_b[l]));

            // ----------------------------------------------------------
            // 2c. Rotary Position Embedding (RoPE)
            //     Works on the 3D tensors directly:
            //       _q       [1, nh,    dh]
            //       _k       [1, nkvh,  dh]
            //       _q_rope  [1, nh,    dh]
            //       _k_rope  [1, nkvh,  dh]
            // ----------------------------------------------------------
            ops::rope(model->_q_rope, model->_q,
                      model->_pos_ids_tensor,
                      meta.theta);
            ops::rope(model->_k_rope, model->_k,
                      model->_pos_ids_tensor,
                      meta.theta);

            // ----------------------------------------------------------
            // 2d. KV-Cache update
            //     Copy the new K/V into the per-layer cache at slot `pos`.
            //     Cache shape: [maxseq, nkvh, dh].
            // ----------------------------------------------------------
            {
                size_t nbytes_k = model->_k_rope->numel()
                                * model->_k_rope->elementSize();
                size_t nbytes_v = model->_v->numel()
                                * model->_v->elementSize();

                auto k_slice = model->_k_cache[l]->slice(0, pos, pos + 1);
                auto v_slice = model->_v_cache[l]->slice(0, pos, pos + 1);

                if (model->_device == LLAISYS_DEVICE_CPU) {
                    std::memcpy(k_slice->data(),
                                model->_k_rope->data(), nbytes_k);
                    std::memcpy(v_slice->data(),
                                model->_v->data(),      nbytes_v);
                } else {
                    auto *api = llaisys::core::context().runtime().api();
                    api->memcpy_sync(k_slice->data(), model->_k_rope->data(),
                                     nbytes_k, LLAISYS_MEMCPY_D2D);
                    api->memcpy_sync(v_slice->data(), model->_v->data(),
                                     nbytes_v, LLAISYS_MEMCPY_D2D);
                }
            }

            // ----------------------------------------------------------
            // 2e. Self-Attention
            //
            //     Q:  [1,       nh, dh]  — current token only
            //     K:  [pos+1, nkvh, dh]  — full cache up to pos
            //     V:  [pos+1, nkvh, dh]
            //     scale = 1 / sqrt(dh)
            //
            //     Output _attn_out: [1, nh, dh]
            // ----------------------------------------------------------
            {
                auto k_full = model->_k_cache[l]->slice(0, 0, pos + 1);
                auto v_full = model->_v_cache[l]->slice(0, 0, pos + 1);

                float scale = 1.0f /
                    std::sqrt(static_cast<float>(meta.dh));

                ops::self_attention(model->_attn_out,
                                    model->_q_rope,
                                    k_full,
                                    v_full,
                                    scale);
            }

            // ----------------------------------------------------------
            // 2f. Output projection
            //     _attn_out [1, nh, dh]  →  view  →  [1, nh*dh]
            //     attn_o_w  [hs, nh*dh]
            //     result    _attn_proj   [1, hs]
            // ----------------------------------------------------------
            ops::linear(model->_attn_proj, model->_attn_flat,
                        unwrap(W.attn_o_w[l]), tensor_t());

            // ----------------------------------------------------------
            // 2g. Residual #1:  _hidden = _hidden + _attn_proj
            // ----------------------------------------------------------
            ops::add(model->_hidden,
                     model->_hidden, model->_attn_proj);

            // ----------------------------------------------------------
            // 2h. RMS Normalization (post_attention_layernorm)
            // ----------------------------------------------------------
            ops::rms_norm(model->_norm_out2,
                          model->_hidden,
                          unwrap(W.mlp_norm_w[l]),
                          meta.epsilon);

            // ----------------------------------------------------------
            // 2i. SwiGLU MLP block
            //
            //     gate       = Linear(norm_out2, mlp_gate_w)    [1, di]
            //     up         = Linear(norm_out2, mlp_up_w)      [1, di]
            //     swiglu_out = swiglu(gate, up)                  [1, di]
            //     mlp_down   = Linear(swiglu_out, mlp_down_w)   [1, hs]
            // ----------------------------------------------------------
            ops::linear(model->_gate_out, model->_norm_out2,
                        unwrap(W.mlp_gate_w[l]), tensor_t());
            ops::linear(model->_up_out, model->_norm_out2,
                        unwrap(W.mlp_up_w[l]),   tensor_t());
            ops::swiglu(model->_swiglu_out,
                        model->_gate_out, model->_up_out);
            ops::linear(model->_mlp_down, model->_swiglu_out,
                        unwrap(W.mlp_down_w[l]), tensor_t());

            // ----------------------------------------------------------
            // 2j. Residual #2:  _hidden = _hidden + _mlp_down
            // ----------------------------------------------------------
            ops::add(model->_hidden,
                     model->_hidden, model->_mlp_down);

        } // end layer loop

        // ==============================================================
        // 3. Final RMS Normalization
        //    _final_norm [1, hs]
        // ==============================================================
        ops::rms_norm(model->_final_norm,
                      model->_hidden,
                      unwrap(W.out_norm_w),
                      meta.epsilon);

        // ==============================================================
        // 4. LM Head  (out_embed [voc, hs])
        //    _logits [1, voc] = _final_norm [1, hs] @ out_embed^T
        // ==============================================================
        ops::linear(model->_logits, model->_final_norm,
                    unwrap(W.out_embed), tensor_t());

        // ==============================================================
        // 5. Argmax  (argmax expects a 1D tensor, so flatten _logits)
        // ==============================================================
        ops::argmax(model->_max_idx, model->_max_val, model->_logits_1d);

        // Read the predicted next-token id back from device memory.
        if (model->_device == LLAISYS_DEVICE_CPU) {
            std::memcpy(&next_token, model->_max_idx->data(), sizeof(int64_t));
        } else {
            auto *api = llaisys::core::context().runtime().api();
            api->memcpy_sync(&next_token, model->_max_idx->data(),
                             sizeof(int64_t), LLAISYS_MEMCPY_D2H);
        }

    } // end token loop

    // Advance the cache pointer by the number of tokens we just processed.
    model->_cached_len += ntoken;

    return next_token;
}

} // namespace llaisys::models
