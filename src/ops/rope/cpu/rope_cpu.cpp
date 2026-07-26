#include "rope_cpu.hpp"

#include <cmath>
#include <vector>

#include "../../../utils.hpp"

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta,
           size_t seqlen, size_t nheads, size_t d) {
    size_t half_d = d / 2;

    std::vector<float> freq(half_d);
    for (size_t j = 0; j < half_d; j++) {
        freq[j] = std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(d));
    }

    for (size_t s = 0; s < seqlen; s++) {
        float pos = static_cast<float>(pos_ids[s]);
        for (size_t j = 0; j < half_d; j++) {
            float angle = pos / freq[j];
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);
            for (size_t h = 0; h < nheads; h++) {
                size_t base_idx = s * nheads * d + h * d;

                float a, b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a = llaisys::utils::cast<float>(in[base_idx + j]);
                    b = llaisys::utils::cast<float>(in[base_idx + half_d + j]);
                } else {
                    a = in[base_idx + j];
                    b = in[base_idx + half_d + j];
                }

                float ra = a * cos_val - b * sin_val;
                float rb = b * cos_val + a * sin_val;

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[base_idx + j] = llaisys::utils::cast<T>(ra);
                    out[base_idx + half_d + j] = llaisys::utils::cast<T>(rb);
                } else {
                    out[base_idx + j] = ra;
                    out[base_idx + half_d + j] = rb;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          float theta, llaisysDataType_t dtype, size_t seqlen, size_t nheads, size_t d) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_<float>(reinterpret_cast<float *>(out),
                            reinterpret_cast<const float *>(in),
                            reinterpret_cast<const int64_t *>(pos_ids),
                            theta, seqlen, nheads, d);
    case LLAISYS_DTYPE_BF16:
        return rope_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out),
                                      reinterpret_cast<const llaisys::bf16_t *>(in),
                                      reinterpret_cast<const int64_t *>(pos_ids),
                                      theta, seqlen, nheads, d);
    case LLAISYS_DTYPE_F16:
        return rope_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out),
                                       reinterpret_cast<const llaisys::fp16_t *>(in),
                                       reinterpret_cast<const int64_t *>(pos_ids),
                                       theta, seqlen, nheads, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
