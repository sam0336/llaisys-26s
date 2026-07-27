#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "Self-Attention: all tensors must be contiguous.");
    ASSERT(attn_val->ndim() == 3, "Self-Attention: attn_val must be 3D [seqlen, nhead, dv].");
    ASSERT(q->ndim() == 3, "Self-Attention: q must be 3D [seqlen, nhead, d].");
    ASSERT(k->ndim() == 3, "Self-Attention: k must be 3D [total_len, nkvhead, d].");
    ASSERT(v->ndim() == 3, "Self-Attention: v must be 3D [total_len, nkvhead, dv].");

    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];

    ASSERT(k->shape()[2] == d, "Self-Attention: k dim d mismatch with q.");
    ASSERT(attn_val->shape()[0] == seqlen, "Self-Attention: attn_val seqlen mismatch.");
    ASSERT(attn_val->shape()[1] == nhead, "Self-Attention: attn_val nhead mismatch.");
    ASSERT(attn_val->shape()[2] == dv, "Self-Attention: attn_val dv mismatch with v.");
    ASSERT(v->shape()[0] == total_len, "Self-Attention: v total_len mismatch with k.");
    ASSERT(v->shape()[1] == nkvhead, "Self-Attention: v nkvhead mismatch with k.");
    ASSERT(total_len >= seqlen, "Self-Attention: total_len must be >= seqlen.");
    ASSERT(nhead % nkvhead == 0, "Self-Attention: nhead must be divisible by nkvhead.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   scale, attn_val->dtype(), seqlen, total_len, nhead, nkvhead, d, dv);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   scale, attn_val->dtype(), seqlen, total_len, nhead, nkvhead, d, dv);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
