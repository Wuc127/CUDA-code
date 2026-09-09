#pragma once

#include <cuda_runtime.h>


// 一个线程块计算 C 中 64 x 64 的区域。
constexpr int SGEMM_V3_BLOCK_TILE_M = 64;
constexpr int SGEMM_V3_BLOCK_TILE_N = 64;
constexpr int SGEMM_V3_BLOCK_TILE_K = 8;

// 每个线程计算一个 4 x 4 的二维输出区域。
constexpr int SGEMM_V3_THREAD_TILE_M = 4;
constexpr int SGEMM_V3_THREAD_TILE_N = 4;
constexpr int SGEMM_V3_NUM_THREADS =
    (SGEMM_V3_BLOCK_TILE_M / SGEMM_V3_THREAD_TILE_M) *
    (SGEMM_V3_BLOCK_TILE_N / SGEMM_V3_THREAD_TILE_N);


// A: M x K
// B: K x N
// C: M x N
// 启动配置：
// threads = dim3(SGEMM_V3_NUM_THREADS)
// blocks  = dim3(ceil(N / BLOCK_TILE_N), ceil(M / BLOCK_TILE_M))
__global__ void sgemm_v3_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);
