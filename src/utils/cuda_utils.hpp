#pragma once

#ifdef __CUDACC__

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

/**
 * @brief Check the result of a CUDA Runtime API call.
 *
 * If the call fails, an exception is thrown with the file, line,
 * error name, and description.
 *
 * Usage:
 *   CUDA_CHECK(cudaMalloc(&ptr, size));
 *   CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
 */
#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t _cuda_check_err = (call);                                               \
        if (_cuda_check_err != cudaSuccess) {                                               \
            std::ostringstream _cuda_check_oss;                                             \
            _cuda_check_oss << "[ERROR] CUDA call failed at " << __FILE__ << ":"            \
                            << __LINE__ << " - " << cudaGetErrorName(_cuda_check_err)       \
                            << ": " << cudaGetErrorString(_cuda_check_err);                 \
            throw std::runtime_error(_cuda_check_oss.str());                                \
        }                                                                                   \
    } while (0)

#endif // __CUDACC__
