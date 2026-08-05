#pragma once

#include "../op.hpp"

#include <cstddef>
#include <cstdint>

namespace llaisys::models::cpu {

/**
 * CPU implementation of Qwen2 model inference.
 *
 * @param model   Pointer to the model instance (holds meta, weights, cache, intermediates).
 * @param token_ids  Array of token ids (host memory).
 * @param ntoken     Number of tokens in this step.
 * @return           The predicted next token id.
 */
int64_t qwen2_infer(Qwen2Model *model, const int64_t *token_ids, size_t ntoken);

} // namespace llaisys::models::cpu
