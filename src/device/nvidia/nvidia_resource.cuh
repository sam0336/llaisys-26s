#pragma once

#include "../device_resource.hpp"

#include <cublas_v2.h>

namespace llaisys::device::nvidia {

class Resource : public llaisys::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource();

    /** @brief cuBLAS handle for this device. */
    cublasHandle_t cublas_handle() const { return _cublas_handle; }

private:
    cublasHandle_t _cublas_handle = nullptr;
};

} // namespace llaisys::device::nvidia
