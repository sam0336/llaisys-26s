#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::musa {

/**
 * @brief Embedding lookup on MUSA GPU.
 *
 * Copies rows from `weight` indexed by `index` into `out`.
 *
 * @param out           Output tensor data (device pointer), 2-D [numel, embedding_dim].
 * @param index         Row indices (device pointer), 1-D int64 [numel].
 * @param weight        Embedding weight matrix (device pointer), 2-D [vocab_size, embedding_dim].
 * @param dtype         Data type of `out` and `weight`.
 * @param numel         Number of rows to look up (length of `index`).
 * @param embedding_dim Size of each embedding vector.
 */
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t dtype, size_t numel, size_t embedding_dim);

} // namespace llaisys::ops::musa
