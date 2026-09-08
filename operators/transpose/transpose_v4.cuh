#pragma once

#include <cuda_runtime.h>


constexpr int TRANSPOSE_V4_TILE_DIM = 64;

// X: M × N
// Y: N × M，使用与 X 独立的存储空间，均按行优先存储
// block: dim3(16, 8)
// grid: dim3((N + 63) / 64, (M + 63) / 64)
__global__ void transpose_v4_kernel(
    const float* X,
    float* Y,
    int M,
    int N
);
