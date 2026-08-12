#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::musa {

/**
 * @brief Linear transformation on MUSA GPU:  Y = X * W^T + bias.
 *
 * Uses muBLAS for the matrix multiplication.
 *
 * @param out    Output tensor data (device pointer), 2-D [m, n].
 * @param in     Input tensor data (device pointer), 2-D [m, k].
 * @param weight Weight matrix data (device pointer), 2-D [n, k].
 * @param bias   Optional bias vector data (device pointer), 1-D [n] or nullptr.
 * @param dtype  Data type.
 * @param m      Batch size (rows of input / output).
 * @param n      Output features (rows of weight).
 * @param k      Input features (cols of input, cols of weight).
 */
void linear(std::byte *out, const std::byte *in,
            const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t m, size_t n, size_t k);

} // namespace llaisys::ops::musa
