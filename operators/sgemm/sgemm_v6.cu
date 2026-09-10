#include "sgemm_v6.cuh"

#include <device_launch_parameters.h>


// 从行主序矩阵中加载连续的 4 个元素。
// 地址不满足 float4 对齐或位于边界时，退回标量加载并补零。
__device__ __forceinline__ float4 load_float4_or_zero(
    const float* data,
    int row,
    int col,
    int num_rows,
    int num_cols
)
{
    float4 values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (row >= num_rows)
    {
        return values;
    }

    if (
        col + SGEMM_V6_VECTOR_WIDTH <= num_cols &&
        num_cols % SGEMM_V6_VECTOR_WIDTH == 0
    )
    {
        return *reinterpret_cast<const float4*>(
            data + row * num_cols + col
        );
    }

    if (col < num_cols)
    {
        values.x = data[row * num_cols + col];
    }
    if (col + 1 < num_cols)
    {
        values.y = data[row * num_cols + col + 1];
    }
    if (col + 2 < num_cols)
    {
        values.z = data[row * num_cols + col + 2];
    }
    if (col + 3 < num_cols)
    {
        values.w = data[row * num_cols + col + 3];
    }

    return values;
}


// Double Buffering：计算当前 tile 时预取下一个 tile
__global__ void sgemm_v6_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{
    __shared__ float A_tiles
        [SGEMM_V6_NUM_BUFFERS]
        [SGEMM_V6_BLOCK_TILE_K]
        [SGEMM_V6_BLOCK_TILE_M + SGEMM_V6_A_PADDING];
    __shared__ float B_tiles
        [SGEMM_V6_NUM_BUFFERS]
        [SGEMM_V6_BLOCK_TILE_K]
        [SGEMM_V6_BLOCK_TILE_N];

    const int thread_id = threadIdx.x;
    const int threads_per_row =
        SGEMM_V6_BLOCK_TILE_N / SGEMM_V6_THREAD_TILE_N;

    const int thread_row = thread_id / threads_per_row;
    const int thread_col = thread_id % threads_per_row;

    const int block_row = blockIdx.y * SGEMM_V6_BLOCK_TILE_M;
    const int block_col = blockIdx.x * SGEMM_V6_BLOCK_TILE_N;

    const int A_vectors_per_row =
        SGEMM_V6_BLOCK_TILE_K / SGEMM_V6_VECTOR_WIDTH;
    const int A_num_vectors =
        SGEMM_V6_BLOCK_TILE_M * A_vectors_per_row;
    const int A_local_row = thread_id / A_vectors_per_row;
    const int A_local_col =
        (thread_id % A_vectors_per_row) * SGEMM_V6_VECTOR_WIDTH;

    const int B_vectors_per_row =
        SGEMM_V6_BLOCK_TILE_N / SGEMM_V6_VECTOR_WIDTH;
    const int B_num_vectors =
        SGEMM_V6_BLOCK_TILE_K * B_vectors_per_row;
    const int B_local_row = thread_id / B_vectors_per_row;
    const int B_local_col =
        (thread_id % B_vectors_per_row) * SGEMM_V6_VECTOR_WIDTH;

    float sums[SGEMM_V6_THREAD_TILE_M][SGEMM_V6_THREAD_TILE_N] = {0.0f};

    const int num_tiles =
        (K + SGEMM_V6_BLOCK_TILE_K - 1) / SGEMM_V6_BLOCK_TILE_K;

    // 先把第一个 tile 放入 buffer 0。
    if (num_tiles > 0)
    {
        if (thread_id < A_num_vectors)
        {
            const float4 values = load_float4_or_zero(
                A,
                block_row + A_local_row,
                A_local_col,
                M,
                K
            );

            A_tiles[0][A_local_col][A_local_row] = values.x;
            A_tiles[0][A_local_col + 1][A_local_row] = values.y;
            A_tiles[0][A_local_col + 2][A_local_row] = values.z;
            A_tiles[0][A_local_col + 3][A_local_row] = values.w;
        }

        if (thread_id < B_num_vectors)
        {
            const float4 values = load_float4_or_zero(
                B,
                B_local_row,
                block_col + B_local_col,
                K,
                N
            );

            B_tiles[0][B_local_row][B_local_col] = values.x;
            B_tiles[0][B_local_row][B_local_col + 1] = values.y;
            B_tiles[0][B_local_row][B_local_col + 2] = values.z;
            B_tiles[0][B_local_row][B_local_col + 3] = values.w;
        }
    }

    __syncthreads();

    int current_buffer = 0;

    for (int tile = 0; tile < num_tiles; ++tile)
    {
        const int next_tile = tile + 1;
        const int next_buffer = current_buffer ^ 1;

        float4 next_A = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 next_B = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // 普通全局内存加载先进入寄存器，不使用 Ampere 的 cp.async。
        if (next_tile < num_tiles)
        {
            if (thread_id < A_num_vectors)
            {
                next_A = load_float4_or_zero(
                    A,
                    block_row + A_local_row,
                    next_tile * SGEMM_V6_BLOCK_TILE_K + A_local_col,
                    M,
                    K
                );
            }

            if (thread_id < B_num_vectors)
            {
                next_B = load_float4_or_zero(
                    B,
                    next_tile * SGEMM_V6_BLOCK_TILE_K + B_local_row,
                    block_col + B_local_col,
                    K,
                    N
                );
            }
        }

        // 使用当前共享内存 buffer 完成 4 x 4 外积累加。
#pragma unroll
        for (int k = 0; k < SGEMM_V6_BLOCK_TILE_K; ++k)
        {
            float A_values[SGEMM_V6_THREAD_TILE_M];
            float B_values[SGEMM_V6_THREAD_TILE_N];

#pragma unroll
            for (int row = 0; row < SGEMM_V6_THREAD_TILE_M; ++row)
            {
                const int A_row =
                    thread_row * SGEMM_V6_THREAD_TILE_M + row;
                A_values[row] = A_tiles[current_buffer][k][A_row];
            }

#pragma unroll
            for (int col = 0; col < SGEMM_V6_THREAD_TILE_N; ++col)
            {
                const int B_col =
                    thread_col * SGEMM_V6_THREAD_TILE_N + col;
                B_values[col] = B_tiles[current_buffer][k][B_col];
            }

#pragma unroll
            for (int row = 0; row < SGEMM_V6_THREAD_TILE_M; ++row)
            {
#pragma unroll
                for (int col = 0; col < SGEMM_V6_THREAD_TILE_N; ++col)
                {
                    sums[row][col] += A_values[row] * B_values[col];
                }
            }
        }

        // 把已经预取到寄存器的下一 tile 写入另一个共享内存 buffer。
        if (next_tile < num_tiles)
        {
            if (thread_id < A_num_vectors)
            {
                A_tiles[next_buffer][A_local_col][A_local_row] = next_A.x;
                A_tiles[next_buffer][A_local_col + 1][A_local_row] = next_A.y;
                A_tiles[next_buffer][A_local_col + 2][A_local_row] = next_A.z;
                A_tiles[next_buffer][A_local_col + 3][A_local_row] = next_A.w;
            }

            if (thread_id < B_num_vectors)
            {
                B_tiles[next_buffer][B_local_row][B_local_col] = next_B.x;
                B_tiles[next_buffer][B_local_row][B_local_col + 1] = next_B.y;
                B_tiles[next_buffer][B_local_row][B_local_col + 2] = next_B.z;
                B_tiles[next_buffer][B_local_row][B_local_col + 3] = next_B.w;
            }
        }

        __syncthreads();
        current_buffer = next_buffer;
    }

    // 每行包含 4 个连续结果，满足对齐时使用 float4 写回。
#pragma unroll
    for (int row = 0; row < SGEMM_V6_THREAD_TILE_M; ++row)
    {
        const int global_row =
            block_row + thread_row * SGEMM_V6_THREAD_TILE_M + row;
        const int global_col =
            block_col + thread_col * SGEMM_V6_THREAD_TILE_N;

        if (
            global_row < M &&
            global_col + SGEMM_V6_VECTOR_WIDTH <= N &&
            N % SGEMM_V6_VECTOR_WIDTH == 0
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
            for (int col = 0; col < SGEMM_V6_THREAD_TILE_N; ++col)
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
