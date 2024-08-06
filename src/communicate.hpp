#pragma once

#include <array>
#include <chrono>
#include <cstdio>
#include <iostream>

#include <nccl.h>
#include <thread>

#include "check_error.h"

enum class SendOrRecv {
    Send = 0,
    Recv = 1
};

std::array<int, 24> static_int_arr;

void CUDART_CB sleepCB(cudaStream_t stream, cudaError_t status,
                       void *userData) {
    int nth = *(int*)userData;
    if (status != cudaSuccess) {
        std::cerr << "sleep callback error for time"
                  << nth
                  << " with error message"
                  << cudaGetErrorString(status)
                  << std::endl;
    }    
    // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::cout << nth << "-th callback" << std::endl;
}

cudaEvent_t thread_main(
    SendOrRecv p2p_case, 
    float* buf, const int count, int peer,
    ncclComm_t comm, cudaStream_t stream)
{
    CUDA_CHECK(cudaSetDevice(int(p2p_case)));
    cudaEvent_t event;
    cudaEventCreate(&event);
    if (p2p_case == SendOrRecv::Send) {
        for (int i = 0; i < 24; ++i) {
            NCCL_CHECK(ncclSend(buf, count, ncclFloat32, peer, comm, stream));
        }
    } else if (p2p_case == SendOrRecv::Recv) {
        for (int i = 0; i < 24; ++i) {
            NCCL_CHECK(ncclRecv(buf, count, ncclFloat32, peer, comm, stream));
            static_int_arr[i] = i;
            cudaStreamAddCallback(stream, sleepCB, (void*)(static_int_arr.data()+i), 0);
        }
    } else {
        std::cerr << "bad argument for send or recv" << std::endl;
    }

    cudaEventRecord(event, stream);
    return event;
}