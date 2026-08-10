#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "qwen2_infer.hpp"

namespace llaisys::models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta *meta,
                       llaisysDeviceType_t device,
                       int *device_ids,
                       int ndevice)
    : _meta(*meta)
    , _device(device)
    , _device_id((ndevice > 0 && device_ids) ? device_ids[0] : 0)
    , _cached_len(0) {

    // Allocate KV-Cache tensors (one per layer)
    _k_cache.reserve(_meta.nlayer);
    _v_cache.reserve(_meta.nlayer);

    for (size_t l = 0; l < _meta.nlayer; l++) {
        _k_cache.push_back(Tensor::create(
            {_meta.maxseq, _meta.nkvh, _meta.dh},
            _meta.dtype, _device, _device_id));
        _v_cache.push_back(Tensor::create(
            {_meta.maxseq, _meta.nkvh, _meta.dh},
            _meta.dtype, _device, _device_id));
    }

    // Allocate reusable intermediate tensors
    _allocate_intermediates();
}

void Qwen2Model::_allocate_intermediates() {
    auto dt = _meta.dtype;
    auto dev = _device;
    auto did = _device_id;
    size_t hs = _meta.hs;
    size_t nh = _meta.nh;
    size_t nkvh = _meta.nkvh;
    size_t dh = _meta.dh;
    size_t di = _meta.di;
    size_t voc = _meta.voc;

    // Token / position ids: 1D I64
    _token_ids_tensor = Tensor::create({1}, LLAISYS_DTYPE_I64, dev, did);
    _pos_ids_tensor   = Tensor::create({1}, LLAISYS_DTYPE_I64, dev, did);

    // Hidden state: [1, hs]
    _hidden    = Tensor::create({1, hs}, dt, dev, did);

    // RMS norm outputs: [1, hs]
    _norm_out  = Tensor::create({1, hs}, dt, dev, did);
    _norm_out2 = Tensor::create({1, hs}, dt, dev, did);

    // Q/K/V projections
    _q = Tensor::create({1, nh, dh},  dt, dev, did);
    _k = Tensor::create({1, nkvh, dh}, dt, dev, did);
    _v = Tensor::create({1, nkvh, dh}, dt, dev, did);

    // RoPE outputs
    _q_rope = Tensor::create({1, nh, dh},  dt, dev, did);
    _k_rope = Tensor::create({1, nkvh, dh}, dt, dev, did);

    // Attention output: [1, nh, dh]
    _attn_out  = Tensor::create({1, nh, dh}, dt, dev, did);

    // Attention output projection: [1, hs]
    _attn_proj = Tensor::create({1, hs}, dt, dev, did);

    // SwiGLU intermediates: [1, di]
    _gate_out   = Tensor::create({1, di}, dt, dev, did);
    _up_out     = Tensor::create({1, di}, dt, dev, did);
    _swiglu_out = Tensor::create({1, di}, dt, dev, did);

    // MLP down projection: [1, hs]
    _mlp_down = Tensor::create({1, hs}, dt, dev, did);

    // Final RMS norm: [1, hs]
    _final_norm = Tensor::create({1, hs}, dt, dev, did);

    // Logits: [1, voc]
    _logits = Tensor::create({1, voc}, dt, dev, did);

    // Argmax outputs
    _max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, dev, did);
    _max_val = Tensor::create({1}, dt, dev, did);

    // Pre-allocate flattened 2D views for linear ops.
    _q_flat    = _q->view({1, nh * dh});
    _k_flat    = _k->view({1, nkvh * dh});
    _v_flat    = _v->view({1, nkvh * dh});
    _attn_flat = _attn_out->view({1, nh * dh});
    _logits_1d = _logits->view({voc});
}

void Qwen2Model::_validate_weights() const {
    // Verify that the model weights have been populated before inference.
    
    ASSERT(_weights.in_embed, "Weights not loaded: in_embed is null.");
    ASSERT(_weights.out_embed, "Weights not loaded: out_embed is null.");
    ASSERT(_weights.out_norm_w, "Weights not loaded: out_norm_w is null.");

    ASSERT(_weights.attn_norm_w, "Weights not loaded: attn_norm_w array is null.");
    ASSERT(_weights.attn_q_w,    "Weights not loaded: attn_q_w array is null.");
    ASSERT(_weights.attn_k_w,    "Weights not loaded: attn_k_w array is null.");
    ASSERT(_weights.attn_v_w,    "Weights not loaded: attn_v_w array is null.");
    ASSERT(_weights.attn_o_w,    "Weights not loaded: attn_o_w array is null.");
    ASSERT(_weights.mlp_norm_w,  "Weights not loaded: mlp_norm_w array is null.");
    ASSERT(_weights.mlp_gate_w,  "Weights not loaded: mlp_gate_w array is null.");
    ASSERT(_weights.mlp_up_w,    "Weights not loaded: mlp_up_w array is null.");
    ASSERT(_weights.mlp_down_w,  "Weights not loaded: mlp_down_w array is null.");

    for (size_t l = 0; l < _meta.nlayer; l++) {
        ASSERT(_weights.attn_norm_w[l], "Weights not loaded: an attn_norm_w[l] is null.");
        ASSERT(_weights.attn_q_w[l],    "Weights not loaded: an attn_q_w[l] is null.");
        ASSERT(_weights.attn_k_w[l],    "Weights not loaded: an attn_k_w[l] is null.");
        ASSERT(_weights.attn_v_w[l],    "Weights not loaded: an attn_v_w[l] is null.");
        ASSERT(_weights.attn_o_w[l],    "Weights not loaded: an attn_o_w[l] is null.");
        ASSERT(_weights.mlp_norm_w[l],  "Weights not loaded: an mlp_norm_w[l] is null.");
        ASSERT(_weights.mlp_gate_w[l],  "Weights not loaded: an mlp_gate_w[l] is null.");
        ASSERT(_weights.mlp_up_w[l],    "Weights not loaded: an mlp_up_w[l] is null.");
        ASSERT(_weights.mlp_down_w[l],  "Weights not loaded: an mlp_down_w[l] is null.");
    }
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    _validate_weights();

    // Set the correct device before inference
    llaisys::core::context().setDevice(_device, _device_id);

    switch (_device) {
    case LLAISYS_DEVICE_CPU:
        return qwen2_infer(this, token_ids, ntoken);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return qwen2_infer(this, token_ids, ntoken);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::models
