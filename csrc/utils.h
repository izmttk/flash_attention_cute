#pragma once

#include <cuda.h>
#include <tuple>
#include <cstdio>

namespace flash_attention {

#define CUDA_ERROR_CHECK(condition)                                            \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      printf("CUDA_CHECK error in line %d of file %s \
              : %s \n",                                                        \
             __LINE__, __FILE__, cudaGetErrorString(error));                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

inline int get_current_device() {
    int device;
    CUDA_ERROR_CHECK(cudaGetDevice(&device));
    return device;
}

inline std::tuple<int, int> get_compute_capability(int device) {
    int cc_major, cc_minor;
    CUDA_ERROR_CHECK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_ERROR_CHECK(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device));
    return {cc_major, cc_minor};
}

} // namespace flash_attention