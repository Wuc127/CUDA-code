#pragma once

#include <cuda_runtime.h>


// 一个线程块计算 C 中 32 x 32 的区域。
constexpr int SGEMM_V2_BLOCK_TILE_M = 32;
constexpr int SGEMM_V2_BLOCK_TILE_N = 32;
constexpr int SGEMM_V2_BLOCK_TILE_K = 8;

// 每个线程计算同一列中连续的 4 个输出元素。
constexpr int SGEMM_V2_THREAD_TILE_M = 4;
constexpr int SGEMM_V2_NUM_THREADS =
    (SGEMM_V2_BLOCK_TILE_M / SGEMM_V2_THREAD_TILE_M) *
    SGEMM_V2_BLOCK_TILE_N;


// A: M x K
// B: K x N
// C: M x N
// 启动配置：
// threads = dim3(SGEMM_V2_NUM_THREADS)
// blocks  = dim3(ceil(N / BLOCK_TILE_N), ceil(M / BLOCK_TILE_M))
__global__ void sgemm_v2_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);
