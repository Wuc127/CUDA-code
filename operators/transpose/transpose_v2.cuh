#pragma once

#include <cuda_runtime.h>


constexpr int TRANSPOSE_V2_TILE_DIM = 32;

// X: M × N
// Y: N × M，使用与 X 独立的存储空间，均按行优先存储
// block: dim3(32, 32)
// grid: dim3((N + 31) / 32, (M + 31) / 32)
__global__ void transpose_v2_kernel(
    const float* X,
    float* Y,
    int M,
    int N
);
