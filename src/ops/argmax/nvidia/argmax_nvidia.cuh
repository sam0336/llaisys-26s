#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::nvidia {

/**
 * @brief Argmax on NVIDIA GPU — finds the index and value of the maximum element.
 *
 * @param max_idx  Output: index of the max element (device pointer, int64, 1 element).
 * @param max_val  Output: value of the max element (device pointer, 1 element).
 * @param vals     Input tensor data (device pointer), 1-D.
 * @param dtype    Data type of `max_val` and `vals`.
 * @param numel    Number of elements in `vals`.
 */
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t dtype, size_t numel);

} // namespace llaisys::ops::nvidia
