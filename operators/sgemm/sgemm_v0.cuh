#pragma once

#include <cuda_runtime.h>


// A: M × K
// B: K × N
// C: M × N
__global__ void sgemm_v0_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);