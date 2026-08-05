#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"
#include "llaisys_model.hpp"

#include "../models/qwen2/op.hpp"

#include <memory>

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {
        auto model = std::make_shared<llaisys::models::Qwen2Model>(
            meta, device, device_ids, ndevice);
        auto *wrapper = new LlaisysQwen2Model{model};
        return wrapper;
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        return model->model->weights();
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model,
                                    int64_t *token_ids, size_t ntoken) {
        return model->model->infer(token_ids, ntoken);
    }
}
