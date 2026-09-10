#pragma once

#include <cuda_runtime.h>


constexpr int SGEMM_V7_BLOCK_TILE_M = 128;
constexpr int SGEMM_V7_BLOCK_TILE_N = 128;
constexpr int SGEMM_V7_BLOCK_TILE_K = 8;

constexpr int SGEMM_V7_WARP_TILE_M = 64;
constexpr int SGEMM_V7_WARP_TILE_N = 32;

constexpr int SGEMM_V7_THREAD_TILE_M = 8;
constexpr int SGEMM_V7_THREAD_TILE_N = 8;

constexpr int SGEMM_V7_VECTOR_WIDTH = 4;
constexpr int SGEMM_V7_SHARED_PADDING = 1;
constexpr int SGEMM_V7_NUM_BUFFERS = 2;
constexpr int SGEMM_V7_WARP_SIZE = 32;

constexpr int SGEMM_V7_WARPS_M =
    SGEMM_V7_BLOCK_TILE_M / SGEMM_V7_WARP_TILE_M;
constexpr int SGEMM_V7_WARPS_N =
    SGEMM_V7_BLOCK_TILE_N / SGEMM_V7_WARP_TILE_N;
constexpr int SGEMM_V7_NUM_WARPS =
    SGEMM_V7_WARPS_M * SGEMM_V7_WARPS_N;
constexpr int SGEMM_V7_NUM_THREADS =
    SGEMM_V7_NUM_WARPS * SGEMM_V7_WARP_SIZE;


// A: M x K
// B: K x N
// C: M x N
// 启动配置：
// threads = dim3(SGEMM_V7_NUM_THREADS)
// blocks  = dim3(ceil(N / BLOCK_TILE_N), ceil(M / BLOCK_TILE_M))
__global__ void sgemm_v7_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);
