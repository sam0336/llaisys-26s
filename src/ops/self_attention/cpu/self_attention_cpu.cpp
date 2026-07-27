#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     float scale, size_t seqlen, size_t total_len, size_t nhead,
                     size_t nkvhead, size_t d, size_t dv) {
    size_t nrep = nhead / nkvhead;
    std::vector<float> scores(total_len);

    for (size_t h = 0; h < nhead; h++) {
        size_t kh = h / nrep;

        for (size_t i = 0; i < seqlen; i++) {
            float max_score = -INFINITY;

            for (size_t j = 0; j < total_len; j++) {
                if (j > i + (total_len - seqlen)) {
                    scores[j] = -INFINITY;
                    continue;
                }

                float dot = 0.0f;
                for (size_t dim = 0; dim < d; dim++) {
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        float qv = llaisys::utils::cast<float>(q[i * nhead * d + h * d + dim]);
                        float kval = llaisys::utils::cast<float>(k[j * nkvhead * d + kh * d + dim]);
                        dot += qv * kval;
                    } else {
                        dot += q[i * nhead * d + h * d + dim] *
                               k[j * nkvhead * d + kh * d + dim];
                    }
                }
                dot *= scale;
                scores[j] = dot;
                if (dot > max_score) max_score = dot;
            }

            float sum_exp = 0.0f;
            for (size_t j = 0; j < total_len; j++) {
                float exp_val = std::isinf(scores[j]) ? 0.0f : std::exp(scores[j] - max_score);
                scores[j] = exp_val; 
                sum_exp += exp_val;
            }

            for (size_t dim = 0; dim < dv; dim++) {
                float val = 0.0f;
                for (size_t j = 0; j < total_len; j++) {
                    if (scores[j] > 0.0f) {
                        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                            float vv = llaisys::utils::cast<float>(v[j * nkvhead * dv + kh * dv + dim]);
                            val += scores[j] * vv;
                        } else {
                            val += scores[j] * v[j * nkvhead * dv + kh * dv + dim];
                        }
                    }
                }
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    attn_val[i * nhead * dv + h * dv + dim] = llaisys::utils::cast<T>(val / sum_exp);
                } else {
                    attn_val[i * nhead * dv + h * dv + dim] = val / sum_exp;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    float scale, llaisysDataType_t dtype,
                    size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_<float>(reinterpret_cast<float *>(attn_val),
                                       reinterpret_cast<const float *>(q),
                                       reinterpret_cast<const float *>(k),
                                       reinterpret_cast<const float *>(v),
                                       scale, seqlen, total_len, nhead, nkvhead, d, dv);
    case LLAISYS_DTYPE_BF16:
        return self_attention_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                                                 reinterpret_cast<const llaisys::bf16_t *>(q),
                                                 reinterpret_cast<const llaisys::bf16_t *>(k),
                                                 reinterpret_cast<const llaisys::bf16_t *>(v),
                                                 scale, seqlen, total_len, nhead, nkvhead, d, dv);
    case LLAISYS_DTYPE_F16:
        return self_attention_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                                                 reinterpret_cast<const llaisys::fp16_t *>(q),
                                                 reinterpret_cast<const llaisys::fp16_t *>(k),
                                                 reinterpret_cast<const llaisys::fp16_t *>(v),
                                                 scale, seqlen, total_len, nhead, nkvhead, d, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
