#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t numel, size_t embedding_dim) {
    for (size_t i = 0; i < numel; i++) {
        int64_t idx = index[i];
        std::memcpy(&out[i * embedding_dim], &weight[idx * embedding_dim], embedding_dim * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t dtype, size_t numel, size_t embedding_dim) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index),
                                 reinterpret_cast<const float *>(weight), numel, embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index),
                                           reinterpret_cast<const llaisys::bf16_t *>(weight), numel, embedding_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index),
                                           reinterpret_cast<const llaisys::fp16_t *>(weight), numel, embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }                        
}
} // namespace llaisys::ops::cpu
