#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t n, size_t k) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                float sum = 0.0f;
                for (size_t kk = 0; kk < k; kk++) {
                    sum += llaisys::utils::cast<float>(in[i * k + kk]) * llaisys::utils::cast<float>(weight[j * k + kk]);
                }
                if (bias) sum += llaisys::utils::cast<float>(bias[j]);
                out[i * n + j] = llaisys::utils::cast<T>(sum);
            }
        }
    } else {
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                float sum = 0.0f;
                for (size_t kk = 0; kk < k; kk++) {
                    sum += in[i * k + kk] * weight[j * k + kk];
                }
                if (bias) sum += bias[j];
                out[i * n + j] = sum;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t dtype, size_t m, size_t n, size_t k) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                              reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), m, n, k);
    case LLAISYS_DTYPE_BF16:
        return linear_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                                        reinterpret_cast<const llaisys::bf16_t *>(weight),
                                        reinterpret_cast<const llaisys::bf16_t *>(bias), m, n, k);
    case LLAISYS_DTYPE_F16:
        return linear_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                                        reinterpret_cast<const llaisys::fp16_t *>(weight),
                                        reinterpret_cast<const llaisys::fp16_t *>(bias), m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
