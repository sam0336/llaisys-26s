#include "../runtime_api.hpp"
#include "../../utils/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <cstdio>

namespace llaisys::device::nvidia {

namespace {

/**
 * @brief Map llaisysMemcpyKind_t to cudaMemcpyKind.
 *
 * The two enums intentionally share the same numeric values:
 *   LLAISYS_MEMCPY_H2H = 0  <->  cudaMemcpyHostToHost   = 0
 *   LLAISYS_MEMCPY_H2D = 1  <->  cudaMemcpyHostToDevice = 1
 *   LLAISYS_MEMCPY_D2H = 2  <->  cudaMemcpyDeviceToHost = 2
 *   LLAISYS_MEMCPY_D2D = 3  <->  cudaMemcpyDeviceToDevice = 3
 */
inline cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    return static_cast<cudaMemcpyKind>(kind);
}

/**
 * @brief Run a CUDA call; if the error is benign (destroying nullptr, etc.),
 *        silently ignore it.  Otherwise throw.
 */
inline void cudaCheckBenign(cudaError_t err, const char *file, int line) {
    if (err == cudaSuccess)
        return;
    if (err == cudaErrorInvalidResourceHandle || err == cudaErrorInvalidDevicePointer
        || err == cudaErrorInvalidValue) {
#ifndef NDEBUG
        std::fprintf(stderr, "[WARNING] CUDA benign error at %s:%d - %s: %s\n",
                     file, line, cudaGetErrorName(err), cudaGetErrorString(err));
#endif
        return;
    }
    std::ostringstream oss;
    oss << "[ERROR] CUDA call failed at " << file << ":" << line << " - "
        << cudaGetErrorName(err) << ": " << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
}

} // anonymous namespace

namespace runtime_api {

// ---- Device ----

int getDeviceCount() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device) {
    CUDA_CHECK(cudaSetDevice(device));
}

void deviceSynchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
}

// ---- Stream ----

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return static_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    // Destroying the null/default stream or an already-destroyed stream
    // is not a hard error — silently ignore.
    cudaError_t err = cudaStreamDestroy(static_cast<cudaStream_t>(stream));
    cudaCheckBenign(err, __FILE__, __LINE__);
}

void streamSynchronize(llaisysStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
    CUDA_CHECK(cudaGetLastError());
}

// ---- Memory allocation ----

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    cudaError_t err = cudaFree(ptr);
    cudaCheckBenign(err, __FILE__, __LINE__);
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    cudaError_t err = cudaFreeHost(ptr);
    cudaCheckBenign(err, __FILE__, __LINE__);
}

// ---- Memory copy ----

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size,
                 llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind),
                               static_cast<cudaStream_t>(stream)));
}

// ---- API function table ----

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync,
};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace llaisys::device::nvidia
