#pragma once

#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "check_error.h"

void sgemm(float* d_a, float* d_b, float* d_c,
          const int m, const int n, const int k)
{
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1., beta = 0.;
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_T,
                             m, n, k, &alpha,
                             d_a, m,
                             d_b, n, &beta,
                             d_c, m));
}

int test_main()
{
    std::vector<float> vec(1024, 1.);
    float* d_a, *d_b, *d_c;
    const int m = 20480, n = 4096, k = 1024;
    CUDA_CHECK(cudaMalloc(&d_a, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, m * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(d_a, vec.data(), k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(d_b, vec.data(), k * sizeof(float), cudaMemcpyHostToDevice));
    sgemm(d_a, d_b, d_c, m, n, k);
    float res;
    CUDA_CHECK(cudaMemcpyAsync(&res, d_c, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(0));
    std::cout << "res=10.24? : " << res << std::endl; 
    return 0;
}