#include "sgemm_v2.cuh"

#include <device_launch_parameters.h>


// 一维寄存器分块：一个线程计算 C 中同一列的多个元素
__global__ void sgemm_v2_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{
    __shared__ float
        A_tile[SGEMM_V2_BLOCK_TILE_M][SGEMM_V2_BLOCK_TILE_K];
    __shared__ float
        B_tile[SGEMM_V2_BLOCK_TILE_K][SGEMM_V2_BLOCK_TILE_N];

    const int thread_id = threadIdx.x;

    // 256 个线程按 8 x 32 排列，用于计算输出 tile。
    const int thread_row = thread_id / SGEMM_V2_BLOCK_TILE_N;
    const int thread_col = thread_id % SGEMM_V2_BLOCK_TILE_N;

    const int block_row = blockIdx.y * SGEMM_V2_BLOCK_TILE_M;
    const int block_col = blockIdx.x * SGEMM_V2_BLOCK_TILE_N;

    // 每个线程使用 4 个寄存器保存累加结果。
    float sums[SGEMM_V2_THREAD_TILE_M] = {0.0f};

    const int num_tiles =
        (K + SGEMM_V2_BLOCK_TILE_K - 1) / SGEMM_V2_BLOCK_TILE_K;

    for (int tile = 0; tile < num_tiles; ++tile)
    {
        // 将一维线程编号分别映射到 A tile 和 B tile。
        const int A_local_row = thread_id / SGEMM_V2_BLOCK_TILE_K;
        const int A_local_col = thread_id % SGEMM_V2_BLOCK_TILE_K;
        const int B_local_row = thread_id / SGEMM_V2_BLOCK_TILE_N;
        const int B_local_col = thread_id % SGEMM_V2_BLOCK_TILE_N;

        const int A_global_row = block_row + A_local_row;
        const int A_global_col =
            tile * SGEMM_V2_BLOCK_TILE_K + A_local_col;
        const int B_global_row =
            tile * SGEMM_V2_BLOCK_TILE_K + B_local_row;
        const int B_global_col = block_col + B_local_col;

        // 每个线程分别加载 A 和 B 的一个元素，越界位置补零。
        A_tile[A_local_row][A_local_col] =
            (A_global_row < M && A_global_col < K)
                ? A[A_global_row * K + A_global_col]
                : 0.0f;

        B_tile[B_local_row][B_local_col] =
            (B_global_row < K && B_global_col < N)
                ? B[B_global_row * N + B_global_col]
                : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < SGEMM_V2_BLOCK_TILE_K; ++k)
        {
            // 当前线程的 4 个输出共用同一个 B 元素。
            const float B_value = B_tile[k][thread_col];

#pragma unroll
            for (int result = 0; result < SGEMM_V2_THREAD_TILE_M; ++result)
            {
                const int A_row =
                    thread_row * SGEMM_V2_THREAD_TILE_M + result;

                sums[result] += A_tile[A_row][k] * B_value;
            }
        }

        // 确保所有线程使用完共享内存后再加载下一个 tile。
        __syncthreads();
    }

    const int col = block_col + thread_col;

#pragma unroll
    for (int result = 0; result < SGEMM_V2_THREAD_TILE_M; ++result)
    {
        const int row =
            block_row + thread_row * SGEMM_V2_THREAD_TILE_M + result;

        if (row < M && col < N)
        {
            C[row * N + col] = sums[result];
        }
    }
}
