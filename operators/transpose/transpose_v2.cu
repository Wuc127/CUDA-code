#include "transpose_v2.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// 在 Shared Memory 中增加一列 Padding，避免转置访问时的 Bank Conflict
__global__ void transpose_v2_kernel(
    const float* X,
    float* Y,
    int M,
    int N
)
{
    const int TILE_DIM = TRANSPOSE_V2_TILE_DIM;
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 连续线程读取输入矩阵中同一行的连续元素
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    if (row < M && col < N)
    {
        tile[ty][tx] = X[static_cast<size_t>(row) * N + col];
    }

    // 所有线程都参与同步，保证整个 Tile 已经写入共享内存
    __syncthreads();

    // 交换分块坐标和共享内存下标，使输出写入也连续
    int output_row = blockIdx.x * TILE_DIM + ty;
    int output_col = blockIdx.y * TILE_DIM + tx;

    if (output_row < N && output_col < M)
    {
        Y[static_cast<size_t>(output_row) * M + output_col] = tile[tx][ty];
    }
}
