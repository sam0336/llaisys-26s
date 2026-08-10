#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous.");
    ASSERT(out->ndim() == 3, "RoPE: out must be 3D [seqlen, nhead, d].");
    ASSERT(in->ndim() == 3, "RoPE: in must be 3D [seqlen, nhead, d].");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D [seqlen].");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64.");

    size_t seqlen = in->shape()[0];
    size_t nheads = in->shape()[1];
    size_t d = in->shape()[2];

    ASSERT(d % 2 == 0, "RoPE: head_dim must be even.");
    ASSERT(out->shape()[0] == seqlen && out->shape()[1] == nheads && out->shape()[2] == d,
           "RoPE: out shape mismatch.");
    ASSERT(pos_ids->shape()[0] == seqlen, "RoPE: pos_ids shape mismatch.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(),
                         seqlen, nheads, d);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(),
                         seqlen, nheads, d);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(),
                            seqlen, nheads, d);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
