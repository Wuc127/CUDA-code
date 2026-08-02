#pragma once

#include <cuda_runtime.h>


constexpr int THREAD_PER_BLOCK = 256;


__global__ void reduce_v3_kernel(
    float* input,
    float* output
);