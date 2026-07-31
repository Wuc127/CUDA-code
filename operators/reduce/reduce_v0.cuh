#pragma once

#include <cuda_runtime.h>

__global__ void reduce_v0_kernel(
    float* input,
    float* output
);