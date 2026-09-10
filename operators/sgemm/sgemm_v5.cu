#include "sgemm_v5.cuh"

#include <device_launch_parameters.h>


// Shared Memory Bank Conflict 优化：转置 A tile 并增加 padding
__global__ void sgemm_v5_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{
    // A tile 转置存储，padding 避免相邻 K 行映射到相同 bank。
    __shared__ float A_tile
        [SGEMM_V5_BLOCK_TILE_K]
        [SGEMM_V5_BLOCK_TILE_M + SGEMM_V5_A_PADDING];
    __shared__ float
        B_tile[SGEMM_V5_BLOCK_TILE_K][SGEMM_V5_BLOCK_TILE_N];

    const int thread_id = threadIdx.x;
    const int threads_per_row =
        SGEMM_V5_BLOCK_TILE_N / SGEMM_V5_THREAD_TILE_N;

    const int thread_row = thread_id / threads_per_row;
    const int thread_col = thread_id % threads_per_row;

    const int block_row = blockIdx.y * SGEMM_V5_BLOCK_TILE_M;
    const int block_col = blockIdx.x * SGEMM_V5_BLOCK_TILE_N;

    float sums[SGEMM_V5_THREAD_TILE_M][SGEMM_V5_THREAD_TILE_N] = {0.0f};

    const int num_tiles =
        (K + SGEMM_V5_BLOCK_TILE_K - 1) / SGEMM_V5_BLOCK_TILE_K;

    for (int tile = 0; tile < num_tiles; ++tile)
    {
        const int A_vectors_per_row =
            SGEMM_V5_BLOCK_TILE_K / SGEMM_V5_VECTOR_WIDTH;
        const int A_num_vectors =
            SGEMM_V5_BLOCK_TILE_M * A_vectors_per_row;

        // 使用 float4 从全局内存加载 A，并转置写入共享内存。
        if (thread_id < A_num_vectors)
        {
            const int local_row = thread_id / A_vectors_per_row;
            const int local_col =
                (thread_id % A_vectors_per_row) * SGEMM_V5_VECTOR_WIDTH;
            const int global_row = block_row + local_row;
            const int global_col =
                tile * SGEMM_V5_BLOCK_TILE_K + local_col;

            if (
                global_row < M &&
                global_col + SGEMM_V5_VECTOR_WIDTH <= K &&
                K % SGEMM_V5_VECTOR_WIDTH == 0
            )
            {
                const float4 values =
                    *reinterpret_cast<const float4*>(
                        A + global_row * K + global_col
                    );

                A_tile[local_col][local_row] = values.x;
                A_tile[local_col + 1][local_row] = values.y;
                A_tile[local_col + 2][local_row] = values.z;
                A_tile[local_col + 3][local_row] = values.w;
            }
            else
            {
#pragma unroll
                for (int offset = 0; offset < SGEMM_V5_VECTOR_WIDTH; ++offset)
                {
                    const int col = global_col + offset;
                    A_tile[local_col + offset][local_row] =
                        (global_row < M && col < K)
                            ? A[global_row * K + col]
                            : 0.0f;
                }
            }
        }

        const int B_vectors_per_row =
            SGEMM_V5_BLOCK_TILE_N / SGEMM_V5_VECTOR_WIDTH;
        const int B_num_vectors =
            SGEMM_V5_BLOCK_TILE_K * B_vectors_per_row;

        // 使用 float4 从全局内存加载 B。
        if (thread_id < B_num_vectors)
        {
            const int local_row = thread_id / B_vectors_per_row;
            const int local_col =
                (thread_id % B_vectors_per_row) * SGEMM_V5_VECTOR_WIDTH;
            const int global_row =
                tile * SGEMM_V5_BLOCK_TILE_K + local_row;
            const int global_col = block_col + local_col;

            if (
                global_row < K &&
                global_col + SGEMM_V5_VECTOR_WIDTH <= N &&
                N % SGEMM_V5_VECTOR_WIDTH == 0
            )
            {
                const float4 values =
                    *reinterpret_cast<const float4*>(
                        B + global_row * N + global_col
                    );

                B_tile[local_row][local_col] = values.x;
                B_tile[local_row][local_col + 1] = values.y;
                B_tile[local_row][local_col + 2] = values.z;
                B_tile[local_row][local_col + 3] = values.w;
            }
            else
            {
#pragma unroll
                for (int offset = 0; offset < SGEMM_V5_VECTOR_WIDTH; ++offset)
                {
                    const int col = global_col + offset;
                    B_tile[local_row][local_col + offset] =
                        (global_row < K && col < N)
                            ? B[global_row * N + col]
                            : 0.0f;
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < SGEMM_V5_BLOCK_TILE_K; ++k)
        {
            float A_values[SGEMM_V5_THREAD_TILE_M];
            float B_values[SGEMM_V5_THREAD_TILE_N];

#pragma unroll
            for (int row = 0; row < SGEMM_V5_THREAD_TILE_M; ++row)
            {
                const int A_row =
                    thread_row * SGEMM_V5_THREAD_TILE_M + row;
                A_values[row] = A_tile[k][A_row];
            }

#pragma unroll
            for (int col = 0; col < SGEMM_V5_THREAD_TILE_N; ++col)
            {
                const int B_col =
                    thread_col * SGEMM_V5_THREAD_TILE_N + col;
                B_values[col] = B_tile[k][B_col];
            }

#pragma unroll
            for (int row = 0; row < SGEMM_V5_THREAD_TILE_M; ++row)
            {
#pragma unroll
                for (int col = 0; col < SGEMM_V5_THREAD_TILE_N; ++col)
                {
                    sums[row][col] += A_values[row] * B_values[col];
                }
            }
        }

        __syncthreads();
    }

    // 每行包含 4 个连续结果，满足对齐时使用 float4 写回。
#pragma unroll
    for (int row = 0; row < SGEMM_V5_THREAD_TILE_M; ++row)
    {
        const int global_row =
            block_row + thread_row * SGEMM_V5_THREAD_TILE_M + row;
        const int global_col =
            block_col + thread_col * SGEMM_V5_THREAD_TILE_N;

        if (
            global_row < M &&
            global_col + SGEMM_V5_VECTOR_WIDTH <= N &&
            N % SGEMM_V5_VECTOR_WIDTH == 0
        )
        {
            float4 values;
            values.x = sums[row][0];
            values.y = sums[row][1];
            values.z = sums[row][2];
            values.w = sums[row][3];

            *reinterpret_cast<float4*>(
                C + global_row * N + global_col
            ) = values;
        }
        else
        {
#pragma unroll
            for (int col = 0; col < SGEMM_V5_THREAD_TILE_N; ++col)
            {
                const int output_col = global_col + col;

                if (global_row < M && output_col < N)
                {
                    C[global_row * N + output_col] = sums[row][col];
                }
            }
        }
    }
}
