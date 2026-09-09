#include "sgemm_v4.cuh"

#include <device_launch_parameters.h>


// 向量化访存：使用 float4 加载 A、B，并写回 C
__global__ void sgemm_v4_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{
    __shared__ float
        A_tile[SGEMM_V4_BLOCK_TILE_M][SGEMM_V4_BLOCK_TILE_K];
    __shared__ float
        B_tile[SGEMM_V4_BLOCK_TILE_K][SGEMM_V4_BLOCK_TILE_N];

    const int thread_id = threadIdx.x;
    const int threads_per_row =
        SGEMM_V4_BLOCK_TILE_N / SGEMM_V4_THREAD_TILE_N;

    const int thread_row = thread_id / threads_per_row;
    const int thread_col = thread_id % threads_per_row;

    const int block_row = blockIdx.y * SGEMM_V4_BLOCK_TILE_M;
    const int block_col = blockIdx.x * SGEMM_V4_BLOCK_TILE_N;

    float sums[SGEMM_V4_THREAD_TILE_M][SGEMM_V4_THREAD_TILE_N] = {0.0f};

    const int num_tiles =
        (K + SGEMM_V4_BLOCK_TILE_K - 1) / SGEMM_V4_BLOCK_TILE_K;

    for (int tile = 0; tile < num_tiles; ++tile)
    {
        const int A_vectors_per_row =
            SGEMM_V4_BLOCK_TILE_K / SGEMM_V4_VECTOR_WIDTH;
        const int A_num_vectors =
            SGEMM_V4_BLOCK_TILE_M * A_vectors_per_row;

        // 前 128 个线程分别加载 A tile 中连续的 4 个元素。
        if (thread_id < A_num_vectors)
        {
            const int local_row = thread_id / A_vectors_per_row;
            const int local_col =
                (thread_id % A_vectors_per_row) * SGEMM_V4_VECTOR_WIDTH;
            const int global_row = block_row + local_row;
            const int global_col =
                tile * SGEMM_V4_BLOCK_TILE_K + local_col;

            if (
                global_row < M &&
                global_col + SGEMM_V4_VECTOR_WIDTH <= K &&
                K % SGEMM_V4_VECTOR_WIDTH == 0
            )
            {
                const float4 values =
                    *reinterpret_cast<const float4*>(
                        A + global_row * K + global_col
                    );

                A_tile[local_row][local_col] = values.x;
                A_tile[local_row][local_col + 1] = values.y;
                A_tile[local_row][local_col + 2] = values.z;
                A_tile[local_row][local_col + 3] = values.w;
            }
            else
            {
#pragma unroll
                for (int offset = 0; offset < SGEMM_V4_VECTOR_WIDTH; ++offset)
                {
                    const int col = global_col + offset;
                    A_tile[local_row][local_col + offset] =
                        (global_row < M && col < K)
                            ? A[global_row * K + col]
                            : 0.0f;
                }
            }
        }

        const int B_vectors_per_row =
            SGEMM_V4_BLOCK_TILE_N / SGEMM_V4_VECTOR_WIDTH;
        const int B_num_vectors =
            SGEMM_V4_BLOCK_TILE_K * B_vectors_per_row;

        // 前 128 个线程分别加载 B tile 中连续的 4 个元素。
        if (thread_id < B_num_vectors)
        {
            const int local_row = thread_id / B_vectors_per_row;
            const int local_col =
                (thread_id % B_vectors_per_row) * SGEMM_V4_VECTOR_WIDTH;
            const int global_row =
                tile * SGEMM_V4_BLOCK_TILE_K + local_row;
            const int global_col = block_col + local_col;

            if (
                global_row < K &&
                global_col + SGEMM_V4_VECTOR_WIDTH <= N &&
                N % SGEMM_V4_VECTOR_WIDTH == 0
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
                for (int offset = 0; offset < SGEMM_V4_VECTOR_WIDTH; ++offset)
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
        for (int k = 0; k < SGEMM_V4_BLOCK_TILE_K; ++k)
        {
            float A_values[SGEMM_V4_THREAD_TILE_M];
            float B_values[SGEMM_V4_THREAD_TILE_N];

#pragma unroll
            for (int row = 0; row < SGEMM_V4_THREAD_TILE_M; ++row)
            {
                const int A_row =
                    thread_row * SGEMM_V4_THREAD_TILE_M + row;
                A_values[row] = A_tile[A_row][k];
            }

#pragma unroll
            for (int col = 0; col < SGEMM_V4_THREAD_TILE_N; ++col)
            {
                const int B_col =
                    thread_col * SGEMM_V4_THREAD_TILE_N + col;
                B_values[col] = B_tile[k][B_col];
            }

#pragma unroll
            for (int row = 0; row < SGEMM_V4_THREAD_TILE_M; ++row)
            {
#pragma unroll
                for (int col = 0; col < SGEMM_V4_THREAD_TILE_N; ++col)
                {
                    sums[row][col] += A_values[row] * B_values[col];
                }
            }
        }

        __syncthreads();
    }

    // 每行包含 4 个连续结果，满足对齐时使用 float4 写回。
#pragma unroll
    for (int row = 0; row < SGEMM_V4_THREAD_TILE_M; ++row)
    {
        const int global_row =
            block_row + thread_row * SGEMM_V4_THREAD_TILE_M + row;
        const int global_col =
            block_col + thread_col * SGEMM_V4_THREAD_TILE_N;

        if (
            global_row < M &&
            global_col + SGEMM_V4_VECTOR_WIDTH <= N &&
            N % SGEMM_V4_VECTOR_WIDTH == 0
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
            for (int col = 0; col < SGEMM_V4_THREAD_TILE_N; ++col)
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
