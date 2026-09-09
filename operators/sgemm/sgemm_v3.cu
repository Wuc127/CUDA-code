#include "sgemm_v3.cuh"

#include <device_launch_parameters.h>


// 二维寄存器分块：一个线程计算 C 中 4 x 4 的区域
__global__ void sgemm_v3_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{
    __shared__ float
        A_tile[SGEMM_V3_BLOCK_TILE_M][SGEMM_V3_BLOCK_TILE_K];
    __shared__ float
        B_tile[SGEMM_V3_BLOCK_TILE_K][SGEMM_V3_BLOCK_TILE_N];

    const int thread_id = threadIdx.x;
    const int threads_per_row =
        SGEMM_V3_BLOCK_TILE_N / SGEMM_V3_THREAD_TILE_N;

    // 256 个线程按 16 x 16 排列，用于计算输出 tile。
    const int thread_row = thread_id / threads_per_row;
    const int thread_col = thread_id % threads_per_row;

    const int block_row = blockIdx.y * SGEMM_V3_BLOCK_TILE_M;
    const int block_col = blockIdx.x * SGEMM_V3_BLOCK_TILE_N;

    // 每个线程使用 4 x 4 个寄存器保存累加结果。
    float sums[SGEMM_V3_THREAD_TILE_M][SGEMM_V3_THREAD_TILE_N] = {0.0f};

    const int num_tiles =
        (K + SGEMM_V3_BLOCK_TILE_K - 1) / SGEMM_V3_BLOCK_TILE_K;

    for (int tile = 0; tile < num_tiles; ++tile)
    {
        // 线程协作加载 A tile，每个线程加载两个元素。
        for (
            int index = thread_id;
            index < SGEMM_V3_BLOCK_TILE_M * SGEMM_V3_BLOCK_TILE_K;
            index += SGEMM_V3_NUM_THREADS
        )
        {
            const int local_row = index / SGEMM_V3_BLOCK_TILE_K;
            const int local_col = index % SGEMM_V3_BLOCK_TILE_K;
            const int global_row = block_row + local_row;
            const int global_col =
                tile * SGEMM_V3_BLOCK_TILE_K + local_col;

            A_tile[local_row][local_col] =
                (global_row < M && global_col < K)
                    ? A[global_row * K + global_col]
                    : 0.0f;
        }

        // 线程协作加载 B tile，每个线程加载两个元素。
        for (
            int index = thread_id;
            index < SGEMM_V3_BLOCK_TILE_K * SGEMM_V3_BLOCK_TILE_N;
            index += SGEMM_V3_NUM_THREADS
        )
        {
            const int local_row = index / SGEMM_V3_BLOCK_TILE_N;
            const int local_col = index % SGEMM_V3_BLOCK_TILE_N;
            const int global_row =
                tile * SGEMM_V3_BLOCK_TILE_K + local_row;
            const int global_col = block_col + local_col;

            B_tile[local_row][local_col] =
                (global_row < K && global_col < N)
                    ? B[global_row * N + global_col]
                    : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < SGEMM_V3_BLOCK_TILE_K; ++k)
        {
            float A_values[SGEMM_V3_THREAD_TILE_M];
            float B_values[SGEMM_V3_THREAD_TILE_N];

#pragma unroll
            for (int row = 0; row < SGEMM_V3_THREAD_TILE_M; ++row)
            {
                const int A_row =
                    thread_row * SGEMM_V3_THREAD_TILE_M + row;
                A_values[row] = A_tile[A_row][k];
            }

#pragma unroll
            for (int col = 0; col < SGEMM_V3_THREAD_TILE_N; ++col)
            {
                const int B_col =
                    thread_col * SGEMM_V3_THREAD_TILE_N + col;
                B_values[col] = B_tile[k][B_col];
            }

            // 使用外积更新当前线程负责的 4 x 4 个结果。
#pragma unroll
            for (int row = 0; row < SGEMM_V3_THREAD_TILE_M; ++row)
            {
#pragma unroll
                for (int col = 0; col < SGEMM_V3_THREAD_TILE_N; ++col)
                {
                    sums[row][col] += A_values[row] * B_values[col];
                }
            }
        }

        // 确保所有线程使用完共享内存后再加载下一个 tile。
        __syncthreads();
    }

    // 将每个线程负责的 4 x 4 个结果写回全局内存。
#pragma unroll
    for (int row = 0; row < SGEMM_V3_THREAD_TILE_M; ++row)
    {
        const int global_row =
            block_row + thread_row * SGEMM_V3_THREAD_TILE_M + row;

#pragma unroll
        for (int col = 0; col < SGEMM_V3_THREAD_TILE_N; ++col)
        {
            const int global_col =
                block_col + thread_col * SGEMM_V3_THREAD_TILE_N + col;

            if (global_row < M && global_col < N)
            {
                C[global_row * N + global_col] = sums[row][col];
            }
        }
    }
}
