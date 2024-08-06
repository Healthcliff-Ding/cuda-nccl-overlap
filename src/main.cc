#include <chrono>
#include <ctime>
#include <iostream>
#include <future>
#include <thread>
#include <vector>
#include <cassert>
#include <mpi.h>

#include <nccl.h>
#include <cuda_runtime.h>

#include "compute.hpp"
#include "communicate.hpp"
#include "check_error.h"

int main(int argc, char* argv[])
{
    // Init MPI and NCCL
    int stat = MPI_Init(&argc, &argv);
    if (stat != MPI_SUCCESS) {
        printf("Failed to init MPI\n");
        return 1;
    }
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    ncclUniqueId comm_id;
    ncclComm_t comm;
    if (rank == 0) {
        ncclGetUniqueId(&comm_id);
    }
    MPI_Bcast(&comm_id, sizeof(comm_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaSetDevice(rank));
    NCCL_CHECK(ncclCommInitRank(&comm, world_size, comm_id, rank));

    // Prepare data
    // sgemm
    std::vector<float> vec(1024, 0.1);
    cublasHandle_t handle;
    float* d_a, *d_b, *d_c;
    const int m = 20480, n = 4096, k = 1024;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaMalloc(&d_a, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, m * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(d_a, vec.data(), k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(d_b, vec.data(), k * sizeof(float), cudaMemcpyHostToDevice));

    float sgemm_res;
    // nccl
    cudaStream_t nccl_stream;
    cudaStreamCreateWithPriority(&nccl_stream, cudaStreamNonBlocking, -5);
    float* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, n * k * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_buf, 1 - rank, n * k * sizeof(float)));

    //! don't forget to initialize chain startup
    CUDA_CHECK(cudaDeviceSynchronize());
    if (rank == int(SendOrRecv::Send)) {
        NCCL_CHECK(ncclSend(d_buf, n * k, ncclFloat, int(SendOrRecv::Recv), comm, nccl_stream));
    } else if (rank == int(SendOrRecv::Recv)) {
        NCCL_CHECK(ncclRecv(d_buf, n * k, ncclFloat, int(SendOrRecv::Send), comm, nccl_stream));
    } else {
        std::cerr << "Invalid rank!" << std::endl;
        exit(1);
    }
    // CUDA_CHECK(cudaStreamSynchronize(nccl_stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Send/Recv data
    if (rank == 0) {
        auto f = std::async(std::launch::async,
                            thread_main,
                            SendOrRecv{rank},
                            d_buf, n * k, 1 - rank,
                            comm, nccl_stream);
        std::cout << "Start to send && compute... ";
        put_now();
        // overlapping compute
        for (int i = 0; i < 1000; ++i) {
            sgemm(d_a, d_b, d_c, m, n, k, handle);
        }
        cudaStreamSynchronize(0);
        std::cout << "Finish computing... ";
        put_now();
        // cudaEvent_t event = f.get();
        // cudaEventSynchronize(event);
        // std::cout << "Finish sending... ";
        // put_now();
    } else if (rank == 1) {
        // std::thread t {thread_main,
        //                SendOrRecv{rank},
        //                d_buf, n * k, 1 - rank,
        //                comm, nccl_stream};
        auto f = std::async(std::launch::async,
                    thread_main,
                    SendOrRecv{rank},
                    d_buf, n * k, 1 - rank,
                    comm, nccl_stream);
        std::cout << "Start to receive... ";                
        put_now();
        // t.join();
        // cudaEvent_t event = f.get();
        // cudaEventSynchronize(event);
        // std::cout << "Finish receiving... ";  
        // put_now();
    } else {
        std::cerr << "invalid rank: " << rank << std::endl;
    }

    MPI_Finalize();
    return 0;
}