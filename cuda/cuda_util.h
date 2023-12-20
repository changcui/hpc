#pragma once

#include "cuda.h"
#include <iostream>
#define CHECK_CUDA_ERROR(val) ::util::check((val), #val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() ::util::checkLast(__FILE__, __LINE__)

namespace util {
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void checkLast(const char *const file, const int line) {
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

} // namespace util
