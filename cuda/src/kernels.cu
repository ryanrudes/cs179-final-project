#include "kernels.cuh"

#include <cuda_runtime.h>

namespace cs179 {

namespace {

__global__ void vector_add_kernel(const float* a, const float* b, float* out, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

}  // namespace

void vector_add(const float* a, const float* b, float* out, std::size_t n) {
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_out = nullptr;

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

}  // namespace cs179
