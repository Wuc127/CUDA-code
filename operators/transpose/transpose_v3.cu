#include "transpose_v3.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>


// 使用 float4 向量化读写，结合 Shared Memory 和 Padding 完成转置
__global__ void transpose_v3_kernel(
    const float* X,
    float* Y,
    int M,
    int N
)
{
    const int TILE_DIM = TRANSPOSE_V3_TILE_DIM;
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // 每个线程负责同一行中连续的 4 个元素
    int tx = threadIdx.x * 4;
    int ty = threadIdx.y;

    {
        int tile_row = ty;
        int row = blockIdx.y * TILE_DIM + tile_row;
        int col = blockIdx.x * TILE_DIM + tx;

        if (row < M && col < N)
        {
            const float* input = X + static_cast<size_t>(row) * N + col;

            // float4 需要 16 字节对齐，同时不能跨过当前行的边界
            if (col + 3 < N &&
                reinterpret_cast<std::uintptr_t>(input) % alignof(float4) == 0)
            {
                float4 value = *reinterpret_cast<const float4*>(input);

                tile[tile_row][tx] = value.x;
                tile[tile_row][tx + 1] = value.y;
                tile[tile_row][tx + 2] = value.z;
                tile[tile_row][tx + 3] = value.w;
            }
            else
            {
                // 行首未对齐或不足 4 个元素时，使用标量读取
                for (int i = 0; i < 4 && col + i < N; i++)
                {
                    tile[tile_row][tx + i] = input[i];
                }
            }
        }
    }

    // 边界线程也必须参与同步
    __syncthreads();

    {
        int tile_row = ty;
        int output_row = blockIdx.x * TILE_DIM + tile_row;
        int output_col = blockIdx.y * TILE_DIM + tx;

        if (output_row < N && output_col < M)
        {
            float* output = Y + static_cast<size_t>(output_row) * M + output_col;

            if (output_col + 3 < M &&
                reinterpret_cast<std::uintptr_t>(output) % alignof(float4) == 0)
            {
                // 从共享内存的同一列收集元素，连续写入输出矩阵的一行
                float4 value;
                value.x = tile[tx][tile_row];
                value.y = tile[tx + 1][tile_row];
                value.z = tile[tx + 2][tile_row];
                value.w = tile[tx + 3][tile_row];

                *reinterpret_cast<float4*>(output) = value;
            }
            else
            {
                // 输出地址未对齐或不足 4 个元素时，使用标量写回
                for (int i = 0; i < 4 && output_col + i < M; i++)
                {
                    output[i] = tile[tx + i][tile_row];
                }
            }
        }
    }
}
