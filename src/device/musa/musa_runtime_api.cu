#include "../runtime_api.hpp"
#include "../../utils/musa_utils.hpp"

#include <musa_runtime.h>
#include <cstdio>

namespace llaisys::device::musa {

namespace {

/**
 * @brief Map llaisysMemcpyKind_t to musaMemcpyKind.
 *
 * The two enums intentionally share the same numeric values:
 *   LLAISYS_MEMCPY_H2H = 0  <->  musaMemcpyHostToHost   = 0
 *   LLAISYS_MEMCPY_H2D = 1  <->  musaMemcpyHostToDevice = 1
 *   LLAISYS_MEMCPY_D2H = 2  <->  musaMemcpyDeviceToHost = 2
 *   LLAISYS_MEMCPY_D2D = 3  <->  musaMemcpyDeviceToDevice = 3
 */
inline musaMemcpyKind toMusaMemcpyKind(llaisysMemcpyKind_t kind) {
    return static_cast<musaMemcpyKind>(kind);
}

/**
 * @brief Run a MUSA call; if the error is benign (destroying nullptr, etc.),
 *        silently ignore it.  Otherwise throw.
 */
inline void musaCheckBenign(musaError_t err, const char *file, int line) {
    if (err == musaSuccess)
        return;
    if (err == musaErrorInvalidResourceHandle || err == musaErrorInvalidDevicePointer
        || err == musaErrorInvalidValue) {
#ifndef NDEBUG
        std::fprintf(stderr, "[WARNING] MUSA benign error at %s:%d - %s: %s\n",
                     file, line, musaGetErrorName(err), musaGetErrorString(err));
#endif
        return;
    }
    std::ostringstream oss;
    oss << "[ERROR] MUSA call failed at " << file << ":" << line << " - "
        << musaGetErrorName(err) << ": " << musaGetErrorString(err);
    throw std::runtime_error(oss.str());
}

} // anonymous namespace

namespace runtime_api {

// ---- Device ----

int getDeviceCount() {
    int count = 0;
    MUSA_CHECK(musaGetDeviceCount(&count));
    return count;
}

void setDevice(int device) {
    MUSA_CHECK(musaSetDevice(device));
}

void deviceSynchronize() {
    MUSA_CHECK(musaDeviceSynchronize());
    MUSA_CHECK(musaGetLastError());
}

// ---- Stream ----

llaisysStream_t createStream() {
    musaStream_t stream = nullptr;
    MUSA_CHECK(musaStreamCreate(&stream));
    return static_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    // Destroying the null/default stream or an already-destroyed stream
    // is not a hard error — silently ignore.
    musaError_t err = musaStreamDestroy(static_cast<musaStream_t>(stream));
    musaCheckBenign(err, __FILE__, __LINE__);
}

void streamSynchronize(llaisysStream_t stream) {
    MUSA_CHECK(musaStreamSynchronize(static_cast<musaStream_t>(stream)));
    MUSA_CHECK(musaGetLastError());
}

// ---- Memory allocation ----

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    MUSA_CHECK(musaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    musaError_t err = musaFree(ptr);
    musaCheckBenign(err, __FILE__, __LINE__);
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    MUSA_CHECK(musaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    musaError_t err = musaFreeHost(ptr);
    musaCheckBenign(err, __FILE__, __LINE__);
}

// ---- Memory copy ----

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    MUSA_CHECK(musaMemcpy(dst, src, size, toMusaMemcpyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size,
                 llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    MUSA_CHECK(musaMemcpyAsync(dst, src, size, toMusaMemcpyKind(kind),
                               static_cast<musaStream_t>(stream)));
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

} // namespace llaisys::device::musa
