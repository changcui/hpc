#include "cuda.h"
#include "cuda_util.h"
#include <array>
#include <functional>
#include <iostream>
#include <memory>

constexpr int M{4096};
constexpr int N{4096};
constexpr int NELEMENTS{M * N};

float measure_performance(std::function<void(cudaStream_t)> bound_function,
                          cudaStream_t stream, int num_repeats = 1000,
                          int num_warmups = 100) {
  cudaEvent_t start, stop;
  float time;

  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  for (int i{0}; i < num_warmups; ++i) {
    bound_function(stream);
  }

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

  CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
  for (int i{0}; i < num_repeats; ++i) {
    bound_function(stream);
  }
  CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  const float latency{time / num_repeats};

  return latency;
}

__global__ void copy_v1(float *src, float *dst) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int in_idx = y * M + x;
  dst[in_idx] = src[in_idx];
}

void copy_v1_test(float *d_src, float *d_dst, cudaStream_t stream) {
  dim3 grid(M / 32, N / 32);
  dim3 block(32, 32);
  copy_v1<<<grid, block, 0, stream>>>(d_src, d_dst);
}

__global__ void copy_v2(float *src, float *dst) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dst[idx] = src[idx];
}

void copy_v2_test(float *d_src, float *d_dst, cudaStream_t stream) {
  dim3 grid(M * N / 256);
  dim3 block(256);
  copy_v2<<<grid, block, 0, stream>>>(d_src, d_dst);
}

// Note: naive implementation
__global__ void transpose_v1(float *src, float *dst) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int in_idx = y * M + x;
  int out_idx = x * N + y;
  dst[out_idx] = src[in_idx];
}

void transpose_v1_test(float *d_src, float *d_dst, cudaStream_t stream) {
  dim3 grid(M / 32, N / 32);
  dim3 block(32, 32);
  transpose_v1<<<grid, block, 0, stream>>>(d_src, d_dst);
}

// Note: share memory implementation w/ bank conflict
__global__ void transpose_v2(float *src, float *dst) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int in_idx = y * M + x;
  __shared__ float buf[32][32];
  buf[threadIdx.y][threadIdx.x] = src[in_idx];
  __syncthreads();
  int xx = blockIdx.y * blockDim.y + threadIdx.x;
  int yy = blockIdx.x * blockDim.x + threadIdx.y;
  dst[yy * M + xx] = buf[threadIdx.x][threadIdx.y];
}

void transpose_v2_test(float *d_src, float *d_dst, cudaStream_t stream) {
  dim3 grid(M / 32, N / 32);
  dim3 block(32, 32);
  transpose_v2<<<grid, block, 0, stream>>>(d_src, d_dst);
}

// Note: share memory implementation w/o bank conflict
__global__ void transpose_v3(float *src, float *dst) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int in_idx = y * M + x;
  __shared__ float buf[32][33];
  buf[threadIdx.y][threadIdx.x] = src[in_idx];
  __syncthreads();
  int xx = blockIdx.y * blockDim.y + threadIdx.x;
  int yy = blockIdx.x * blockDim.x + threadIdx.y;
  dst[yy * M + xx] = buf[threadIdx.x][threadIdx.y];
}

void transpose_v3_test(float *d_src, float *d_dst, cudaStream_t stream) {
  dim3 grid(M / 32, N / 32);
  dim3 block(32, 32);
  transpose_v3<<<grid, block, 0, stream>>>(d_src, d_dst);
}

bool all_close(float *h_src, float *h_dst, float eps = 1e-5) {
  for (int i{0}; i < M; ++i) {
    for (int j{0}; j < N; ++j) {
      if (std::abs(h_src[i * N + j] - h_dst[j * M + i]) > eps) {
        return false;
      }
    }
  }
  return true;
}

int main() {
  float *h_src, *h_dst, *d_src, *d_dst;
  CHECK_CUDA_ERROR(cudaMallocHost(&h_src, sizeof(float) * NELEMENTS));
  CHECK_CUDA_ERROR(cudaMallocHost(&h_dst, sizeof(float) * NELEMENTS));
  CHECK_CUDA_ERROR(cudaMalloc(&d_src, sizeof(float) * NELEMENTS));
  CHECK_CUDA_ERROR(cudaMalloc(&d_dst, sizeof(float) * NELEMENTS));
  for (int i{0}; i < M; ++i) {
    for (int j{0}; j < N; ++j) {
      h_src[i * N + j] = i * N + j;
    }
  }
  CHECK_CUDA_ERROR(cudaMemcpy(d_src, h_src, sizeof(float) * NELEMENTS,
                              cudaMemcpyHostToDevice));
  cudaStream_t stream;
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

  {
    std::function<void(cudaStream_t)> copy_v1_test_warp{
        std::bind(copy_v1_test, d_src, d_dst, std::placeholders::_1)};
    auto latency = measure_performance(copy_v1_test_warp, stream);
    float bandwidth = M * N * sizeof(float) * 2 / latency * 1e-6;
    std::cout << "Bandwidth: " << bandwidth << "GB/s.\n";
  }

  {
    std::function<void(cudaStream_t)> copy_v2_test_warp{
        std::bind(copy_v2_test, d_src, d_dst, std::placeholders::_1)};
    auto latency = measure_performance(copy_v2_test_warp, stream);
    float bandwidth = M * N * sizeof(float) * 2 / latency * 1e-6;
    std::cout << "Bandwidth: " << bandwidth << "GB/s.\n";
  }

  {
    std::function<void(cudaStream_t)> transpose_v1_test_warp{
        std::bind(transpose_v1_test, d_src, d_dst, std::placeholders::_1)};
    auto latency = measure_performance(transpose_v1_test_warp, stream);
    float bandwidth = M * N * sizeof(float) * 2 / latency * 1e-6;
    std::cout << "Bandwidth: " << bandwidth << "GB/s.\n";
  }

  {
    std::function<void(cudaStream_t)> transpose_v2_test_warp{
        std::bind(transpose_v2_test, d_src, d_dst, std::placeholders::_1)};
    auto latency = measure_performance(transpose_v2_test_warp, stream);
    float bandwidth = M * N * sizeof(float) * 2 / latency * 1e-6;
    std::cout << "Bandwidth: " << bandwidth << "GB/s.\n";
  }

  {
    std::function<void(cudaStream_t)> transpose_v3_test_warp{
        std::bind(transpose_v3_test, d_src, d_dst, std::placeholders::_1)};
    auto latency = measure_performance(transpose_v3_test_warp, stream);
    float bandwidth = M * N * sizeof(float) * 2 / latency * 1e-6;
    std::cout << "Bandwidth: " << bandwidth << "GB/s.\n";
  }

  CHECK_CUDA_ERROR(cudaMemcpy(h_dst, d_dst, sizeof(float) * NELEMENTS,
                              cudaMemcpyDeviceToHost));
  if (all_close(h_src, h_dst)) {
    std::cout << "Success.\n";
  } else {
    std::cout << "Failed.\n";
  }

  CHECK_CUDA_ERROR(cudaFreeHost(h_src));
  CHECK_CUDA_ERROR(cudaFreeHost(h_dst));
  CHECK_CUDA_ERROR(cudaFree(d_src));
  CHECK_CUDA_ERROR(cudaFree(d_dst));
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
