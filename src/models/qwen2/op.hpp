#pragma once

#include "../../tensor/tensor.hpp"
#include "llaisys/models/qwen2.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaisys::models {
class Qwen2Model;
int64_t qwen2_infer(Qwen2Model *model, const int64_t *token_ids, size_t ntoken);



class Qwen2Model {
private:
    LlaisysQwen2Meta _meta;
    LlaisysQwen2Weights _weights;
    llaisysDeviceType_t _device;
    int _device_id;

    // ====== KV-Cache ======
    // per-layer key and value caches; shape: [maxseq, nkvh, dh] / [maxseq, nkvh, dv]
    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;
    // current number of tokens stored in the cache
    size_t _cached_len;

    // ====== Reusable intermediate tensors ======
    // Token ids tensor (I64, 1D)
    tensor_t _token_ids_tensor;
    // Position ids tensor (I64, 1D)
    tensor_t _pos_ids_tensor;
    // Current hidden state, shape [seqlen, hs]
    tensor_t _hidden;
    // RMS norm outputs
    tensor_t _norm_out;
    // Q/K/V projection outputs
    tensor_t _q, _k, _v;
    // RoPE outputs
    tensor_t _q_rope, _k_rope;
    // Attention output
    tensor_t _attn_out;
    // Attention output projection
    tensor_t _attn_proj;
    // Second RMS norm output (post_attention_layernorm)
    tensor_t _norm_out2;
    // SwiGLU intermediate
    tensor_t _gate_out, _up_out;
    // SwiGLU output
    tensor_t _swiglu_out;
    // MLP down projection
    tensor_t _mlp_down;
    // Final RMS norm output
    tensor_t _final_norm;
    // Logits [1, voc]
    tensor_t _logits;
    // Argmax output (max_idx, max_val)
    tensor_t _max_idx, _max_val;

    // Pre-allocated flattened views (avoids per-token shared_ptr alloc)
    tensor_t _q_flat, _k_flat, _v_flat;
    tensor_t _attn_flat;
    tensor_t _logits_1d;

    // Pre-allocate all intermediate tensors
    void _allocate_intermediates();

    // Verify that model weights have been loaded before inference
    void _validate_weights() const;

    // Allow the infer implementation to access internal members
    friend int64_t qwen2_infer(Qwen2Model *model, const int64_t *token_ids, size_t ntoken);

public:
    Qwen2Model(const LlaisysQwen2Meta *meta,
               llaisysDeviceType_t device,
               int *device_ids,
               int ndevice);

    ~Qwen2Model() = default;

    LlaisysQwen2Weights *weights() { return &_weights; }

    int64_t infer(const int64_t *token_ids, size_t ntoken);
};

} // namespace llaisys::models
