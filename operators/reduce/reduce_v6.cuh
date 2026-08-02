#pragma once

#include <cuda_runtime.h>


constexpr int THREAD_PER_BLOCK = 256;


__device__ void warpReduce(volatile float *cache, unsigned int tid);


__global__ void reduce_v6_kernel(
    float* input,
    float* output
);