#pragma once

#include <memory>

#include "../models/qwen2/op.hpp"

struct LlaisysQwen2Model {
    std::shared_ptr<llaisys::models::Qwen2Model> model;
};
