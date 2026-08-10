#include "nvidia_resource.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

namespace llaisys::device::nvidia {

Resource::Resource(int device_id)
    : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {
    cublasStatus_t stat = cublasCreate(&_cublas_handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuBLAS handle for device "
                                 + std::to_string(device_id));
    }
}

Resource::~Resource() {
    if (_cublas_handle) {
        cublasDestroy(_cublas_handle);
        _cublas_handle = nullptr;
    }
}

} // namespace llaisys::device::nvidia
