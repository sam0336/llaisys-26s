#pragma once

// MUSA error-checking macro — mirrors cuda_utils.hpp with musa API names.
// Only defined when compiled by the MUSA toolchain (mcc defines __MUSACC__).

#ifdef __MUSACC__

#include <musa_runtime.h>
#include <sstream>
#include <stdexcept>

#define MUSA_CHECK(call)                                                              \
    do {                                                                              \
        musaError_t _musa_check_err = (call);                                         \
        if (_musa_check_err != musaSuccess) {                                         \
            std::ostringstream _musa_check_oss;                                       \
            _musa_check_oss << "[ERROR] MUSA call failed at " << __FILE__ << ":"      \
                            << __LINE__ << " - " << musaGetErrorName(_musa_check_err) \
                            << ": " << musaGetErrorString(_musa_check_err);           \
            throw std::runtime_error(_musa_check_oss.str());                          \
        }                                                                             \
    } while (0)

#endif // __MUSACC__
