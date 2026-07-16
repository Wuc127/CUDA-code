#pragma once

#include <cuda_runtime.h>


__global__ void elementwise_v1_kernel(
    const float* A,
    const float* B,
    float* C,
    int num_elements
);