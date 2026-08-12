#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::musa {

/**
 * @brief RMS Normalization on MUSA GPU.
 *
 * For each row i:  out[i] = weight * in[i] / sqrt(sum(in[i]^2)/d + eps)
 *
 * @param out     Output tensor data (device pointer), 2-D [m, d].
 * @param in      Input tensor data (device pointer), 2-D [m, d].
 * @param weight  Weight vector (device pointer), 1-D [d].
 * @param eps     Epsilon for numerical stability.
 * @param dtype   Data type.
 * @param m       Number of rows.
 * @param d       Number of columns (normalization dimension).
 */
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              float eps, llaisysDataType_t dtype, size_t m, size_t d);

} // namespace llaisys::ops::musa
