#include "rms_norm_cpu.hpp"

#include <cmath>

#include "../../../utils.hpp"

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, size_t m, size_t d) {
    for (size_t i = 0; i < m; i++) {
        float sum_sq = 0.0f;
        for (size_t j = 0; j < d; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float val = llaisys::utils::cast<float>(in[i * d + j]);
                sum_sq += val * val;
            } else {
                sum_sq += in[i * d + j] * in[i * d + j];
            }
        }
        float rms = std::sqrt(sum_sq / d + eps);
        for (size_t j = 0; j < d; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                float val = llaisys::utils::cast<float>(in[i * d + j]);
                float w = llaisys::utils::cast<float>(weight[j]);
                out[i * d + j] = llaisys::utils::cast<T>(val * w / rms);
            } else {
                out[i * d + j] = in[i * d + j] * weight[j] / rms;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, llaisysDataType_t dtype, size_t m, size_t d) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                                reinterpret_cast<const float *>(weight), eps, m, d);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out),
                                          reinterpret_cast<const llaisys::bf16_t *>(in),
                                          reinterpret_cast<const llaisys::bf16_t *>(weight), eps, m, d);
    case LLAISYS_DTYPE_F16:
        return rms_norm_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out),
                                           reinterpret_cast<const llaisys::fp16_t *>(in),
                                           reinterpret_cast<const llaisys::fp16_t *>(weight), eps, m, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu