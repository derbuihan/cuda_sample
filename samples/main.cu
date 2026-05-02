#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>


__global__ void add_kernel(const double* a, const double* b, double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 8;
    const int bytes = n * sizeof(double);


    double h_a[n] = {1, 2, 3, 4, 5, 6, 7, 8};
    double h_b[n] = {10, 20, 30, 40, 50, 60, 70, 80};
    double h_c[n] = {};

    double* d_a = nullptr;
    double* d_b = nullptr;
    double* d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);

    cudaGetLastError();
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for (int i = 0; i < n; ++i) {
        std::printf("%.1f + %.1f = %.1f\n", h_a[i], h_b[i], h_c[i]);
    }

    return 0;
}
