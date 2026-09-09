#pragma once

#include <cuda_runtime.h>


// v1 使用 16 x 16 的线程块和共享内存 tile。
constexpr int SGEMM_V1_TILE_SIZE = 16;


// A: M x K
// B: K x N
// C: M x N
__global__ void sgemm_v1_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);
