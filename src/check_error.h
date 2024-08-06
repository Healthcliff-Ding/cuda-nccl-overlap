#pragma once

#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>

#define CUDA_CHECK(cmd) \
do { \
    cudaError_t result = cmd; \
    if (result != cudaSuccess) { \
        printf("[ERROR] CUDA error %s:%d '%s': (%d) %s\n", __FILE__, __LINE__, #cmd, (int)result, cudaGetErrorString(result)); \
        exit(-1); \
    } \
} while(0)

#define NCCL_CHECK(cmd)                                                                                           \
do {                                                                                                          \
    ncclResult_t result = cmd;                                                                                \
    if (result != ncclSuccess) {                                                                              \
        printf("[ERROR] NCCL error %s:%d '%s' : %s\n", __FILE__, __LINE__, #cmd, ncclGetErrorString(result)); \
        exit(-1);                                                                                             \
    }                                                                                                         \
} while (0)

#define CUBLAS_CHECK(err) \
do { \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << err << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

void put_now()
{
    std::chrono::time_point<std::chrono::system_clock> now;
    std::time_t t_c;
    now = std::chrono::system_clock::now();
    t_c = std::chrono::system_clock::to_time_t(now);
    std::cout << std::put_time(std::localtime(&t_c), "%H:%M:%S")
              << std::endl;
}