#pragma once

#include <cuda_runtime.h>


// X: M × N
// Y: N × M，使用与 X 独立的存储空间
__global__ void transpose_v0_kernel(
    const float* X,
    float* Y,
    int M,
    int N
);
