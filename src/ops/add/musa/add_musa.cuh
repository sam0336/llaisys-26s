#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::musa {

/**
 * @brief Element-wise addition on MUSA GPU.
 *
 * @param c     Output tensor data (device pointer).
 * @param a     First input tensor data (device pointer).
 * @param b     Second input tensor data (device pointer).
 * @param type  Data type of all three tensors.
 * @param numel Number of elements.
 */
void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, size_t numel);

} // namespace llaisys::ops::musa
