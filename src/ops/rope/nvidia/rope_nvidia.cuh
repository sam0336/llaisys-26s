#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::nvidia {

/**
 * @brief Rotary Position Embedding (RoPE) on NVIDIA GPU.
 *
 * @param out      Output tensor (device pointer), 3-D [seqlen, nheads, d].
 * @param in       Input tensor (device pointer), 3-D [seqlen, nheads, d].
 * @param pos_ids  Position ids (device pointer), 1-D int64 [seqlen].
 * @param theta    Base frequency.
 * @param dtype    Data type of `out` and `in`.
 * @param seqlen   Sequence length.
 * @param nheads   Number of heads.
 * @param d        Head dimension (must be even).
 */
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          float theta, llaisysDataType_t dtype,
          size_t seqlen, size_t nheads, size_t d);

} // namespace llaisys::ops::nvidia
