#include "sgemm_v7.cuh"

#include <device_launch_parameters.h>


__device__ __forceinline__ float4 sgemm_v7_load_float4_or_zero(
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
        col + SGEMM_V7_VECTOR_WIDTH <= num_cols &&
        num_cols % SGEMM_V7_VECTOR_WIDTH == 0
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


__device__ __forceinline__ void sgemm_v7_store_float4(
    float* data,
    const float4& values,
    int row,
    int col,
    int num_rows,
    int num_cols
)
{
    if (row >= num_rows)
    {
        return;
    }

    if (
        col + SGEMM_V7_VECTOR_WIDTH <= num_cols &&
        num_cols % SGEMM_V7_VECTOR_WIDTH == 0
    )
    {
        *reinterpret_cast<float4*>(data + row * num_cols + col) = values;
        return;
    }

    if (col < num_cols)
    {
        data[row * num_cols + col] = values.x;
    }
    if (col + 1 < num_cols)
    {
        data[row * num_cols + col + 1] = values.y;
    }
    if (col + 2 < num_cols)
    {
        data[row * num_cols + col + 2] = values.z;
    }
    if (col + 3 < num_cols)
    {
        data[row * num_cols + col + 3] = values.w;
    }
}


// Warp Tiling：每个 warp 负责一个 64 x 32 的输出区域。
__global__ void sgemm_v7_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{
    constexpr int A_groups =
        SGEMM_V7_BLOCK_TILE_M / SGEMM_V7_THREAD_TILE_M;
    constexpr int B_groups =
        SGEMM_V7_BLOCK_TILE_N / SGEMM_V7_THREAD_TILE_N;

    __shared__ float A_tiles
        [SGEMM_V7_NUM_BUFFERS]
        [SGEMM_V7_BLOCK_TILE_K]
        [SGEMM_V7_THREAD_TILE_M]
        [A_groups + SGEMM_V7_SHARED_PADDING];
    __shared__ float B_tiles
        [SGEMM_V7_NUM_BUFFERS]
        [SGEMM_V7_BLOCK_TILE_K]
        [SGEMM_V7_THREAD_TILE_N]
        [B_groups + SGEMM_V7_SHARED_PADDING];

    const int thread_id = threadIdx.x;
    const int warp_id = thread_id / SGEMM_V7_WARP_SIZE;
    const int lane_id = thread_id % SGEMM_V7_WARP_SIZE;

    constexpr int warps_per_row =
        SGEMM_V7_BLOCK_TILE_N / SGEMM_V7_WARP_TILE_N;
    constexpr int warp_threads_per_row =
        SGEMM_V7_WARP_TILE_N / SGEMM_V7_THREAD_TILE_N;

    const int warp_row = warp_id / warps_per_row;
    const int warp_col = warp_id % warps_per_row;
    const int thread_row = lane_id / warp_threads_per_row;
    const int thread_col = lane_id % warp_threads_per_row;

    const int output_row_group =
        warp_row * (SGEMM_V7_WARP_TILE_M / SGEMM_V7_THREAD_TILE_M) +
        thread_row;
    const int output_col_group =
        warp_col * (SGEMM_V7_WARP_TILE_N / SGEMM_V7_THREAD_TILE_N) +
        thread_col;

    const int block_row = blockIdx.y * SGEMM_V7_BLOCK_TILE_M;
    const int block_col = blockIdx.x * SGEMM_V7_BLOCK_TILE_N;

    constexpr int A_vectors_per_row =
        SGEMM_V7_BLOCK_TILE_K / SGEMM_V7_VECTOR_WIDTH;
    const int A_local_row = thread_id / A_vectors_per_row;
    const int A_local_col =
        (thread_id % A_vectors_per_row) * SGEMM_V7_VECTOR_WIDTH;

    constexpr int B_vectors_per_row =
        SGEMM_V7_BLOCK_TILE_N / SGEMM_V7_VECTOR_WIDTH;
    const int B_local_row = thread_id / B_vectors_per_row;
    const int B_local_col =
        (thread_id % B_vectors_per_row) * SGEMM_V7_VECTOR_WIDTH;

    float sums[SGEMM_V7_THREAD_TILE_M][SGEMM_V7_THREAD_TILE_N] = {0.0f};

    const int num_tiles =
        (K + SGEMM_V7_BLOCK_TILE_K - 1) / SGEMM_V7_BLOCK_TILE_K;

    const int A_shared_row = A_local_row % SGEMM_V7_THREAD_TILE_M;
    const int A_shared_group = A_local_row / SGEMM_V7_THREAD_TILE_M;

    if (num_tiles > 0)
    {
        const float4 A_values = sgemm_v7_load_float4_or_zero(
            A,
            block_row + A_local_row,
            A_local_col,
            M,
            K
        );

        A_tiles[0][A_local_col][A_shared_row][A_shared_group] = A_values.x;
        A_tiles[0][A_local_col + 1][A_shared_row][A_shared_group] = A_values.y;
        A_tiles[0][A_local_col + 2][A_shared_row][A_shared_group] = A_values.z;
        A_tiles[0][A_local_col + 3][A_shared_row][A_shared_group] = A_values.w;
    }

    const int B_shared_col = B_local_col % SGEMM_V7_THREAD_TILE_N;
    const int B_shared_group = B_local_col / SGEMM_V7_THREAD_TILE_N;

    if (num_tiles > 0)
    {
        const float4 B_values = sgemm_v7_load_float4_or_zero(
            B,
            B_local_row,
            block_col + B_local_col,
            K,
            N
        );

        B_tiles[0][B_local_row][B_shared_col][B_shared_group] = B_values.x;
        B_tiles[0][B_local_row][B_shared_col + 1][B_shared_group] = B_values.y;
        B_tiles[0][B_local_row][B_shared_col + 2][B_shared_group] = B_values.z;
        B_tiles[0][B_local_row][B_shared_col + 3][B_shared_group] = B_values.w;
    }

    __syncthreads();

    int current_buffer = 0;

    for (int tile = 0; tile < num_tiles; ++tile)
    {
        const int next_tile = tile + 1;
        const int next_buffer = current_buffer ^ 1;

        float4 next_A = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 next_B = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (next_tile < num_tiles)
        {
            next_A = sgemm_v7_load_float4_or_zero(
                A,
                block_row + A_local_row,
                next_tile * SGEMM_V7_BLOCK_TILE_K + A_local_col,
                M,
                K
            );

            next_B = sgemm_v7_load_float4_or_zero(
                B,
                next_tile * SGEMM_V7_BLOCK_TILE_K + B_local_row,
                block_col + B_local_col,
                K,
                N
            );
        }

#pragma unroll
        for (int k = 0; k < SGEMM_V7_BLOCK_TILE_K; ++k)
        {
            float A_values[SGEMM_V7_THREAD_TILE_M];
            float B_values[SGEMM_V7_THREAD_TILE_N];

#pragma unroll
            for (int row = 0; row < SGEMM_V7_THREAD_TILE_M; ++row)
            {
                A_values[row] =
                    A_tiles[current_buffer][k][row][output_row_group];
            }

#pragma unroll
            for (int col = 0; col < SGEMM_V7_THREAD_TILE_N; ++col)
            {
                B_values[col] =
                    B_tiles[current_buffer][k][col][output_col_group];
            }

#pragma unroll
            for (int row = 0; row < SGEMM_V7_THREAD_TILE_M; ++row)
            {
#pragma unroll
                for (int col = 0; col < SGEMM_V7_THREAD_TILE_N; ++col)
                {
                    sums[row][col] += A_values[row] * B_values[col];
                }
            }
        }

        if (next_tile < num_tiles)
        {
            A_tiles[next_buffer][A_local_col][A_shared_row][A_shared_group] =
                next_A.x;
            A_tiles[next_buffer][A_local_col + 1][A_shared_row][A_shared_group] =
                next_A.y;
            A_tiles[next_buffer][A_local_col + 2][A_shared_row][A_shared_group] =
                next_A.z;
            A_tiles[next_buffer][A_local_col + 3][A_shared_row][A_shared_group] =
                next_A.w;

            B_tiles[next_buffer][B_local_row][B_shared_col][B_shared_group] =
                next_B.x;
            B_tiles[next_buffer][B_local_row][B_shared_col + 1][B_shared_group] =
                next_B.y;
            B_tiles[next_buffer][B_local_row][B_shared_col + 2][B_shared_group] =
                next_B.z;
            B_tiles[next_buffer][B_local_row][B_shared_col + 3][B_shared_group] =
                next_B.w;
        }

        __syncthreads();
        current_buffer = next_buffer;
    }

    constexpr int output_vectors =
        SGEMM_V7_THREAD_TILE_N / SGEMM_V7_VECTOR_WIDTH;

#pragma unroll
    for (int row = 0; row < SGEMM_V7_THREAD_TILE_M; ++row)
    {
        const int global_row =
            block_row + output_row_group * SGEMM_V7_THREAD_TILE_M + row;

#pragma unroll
        for (int vector = 0; vector < output_vectors; ++vector)
        {
            const int first_col = vector * SGEMM_V7_VECTOR_WIDTH;
            const int global_col =
                block_col +
                output_col_group * SGEMM_V7_THREAD_TILE_N +
                first_col;

            float4 values;
            values.x = sums[row][first_col];
            values.y = sums[row][first_col + 1];
            values.z = sums[row][first_col + 2];
            values.w = sums[row][first_col + 3];

            sgemm_v7_store_float4(
                C,
                values,
                global_row,
                global_col,
                M,
                N
            );
        }
    }
}
