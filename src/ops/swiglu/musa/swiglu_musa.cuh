#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::musa {

/**
 * @brief SwiGLU activation on MUSA GPU.
 *
 * Computes:  out[i] = up[i] * gate[i] / (1 + exp(-gate[i]))
 *
 * @param out    Output tensor data (device pointer).
 * @param gate   Gate tensor data (device pointer).
 * @param up     Up-projection tensor data (device pointer).
 * @param dtype  Data type of all three tensors.
 * @param numel  Number of elements.
 */
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t dtype, size_t numel);

} // namespace llaisys::ops::musa
