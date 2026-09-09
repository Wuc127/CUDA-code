#include "sgemm_v1.cuh"

#include <device_launch_parameters.h>


// Shared Memory Tiling：一个线程块计算 C 中的一个 tile
__global__ void sgemm_v1_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{
    __shared__ float A_tile[SGEMM_V1_TILE_SIZE][SGEMM_V1_TILE_SIZE];
    __shared__ float B_tile[SGEMM_V1_TILE_SIZE][SGEMM_V1_TILE_SIZE];

    const int local_row = threadIdx.y;
    const int local_col = threadIdx.x;

    // 当前线程负责计算的 C 矩阵行号和列号
    const int row = blockIdx.y * SGEMM_V1_TILE_SIZE + local_row;
    const int col = blockIdx.x * SGEMM_V1_TILE_SIZE + local_col;

    float sum = 0.0f;

    // 沿 K 维依次处理 A 和 B 的 tile
    const int num_tiles =
        (K + SGEMM_V1_TILE_SIZE - 1) / SGEMM_V1_TILE_SIZE;

    for (int tile = 0; tile < num_tiles; ++tile)
    {
        const int A_col = tile * SGEMM_V1_TILE_SIZE + local_col;
        const int B_row = tile * SGEMM_V1_TILE_SIZE + local_row;

        // 每个线程分别加载 A 和 B 的一个元素到共享内存。
        // 边界 tile 中越界的元素补零。
        A_tile[local_row][local_col] =
            (row < M && A_col < K) ? A[row * K + A_col] : 0.0f;

        B_tile[local_row][local_col] =
            (B_row < K && col < N) ? B[B_row * N + col] : 0.0f;

        // 等待当前 tile 的数据全部加载完成
        __syncthreads();

#pragma unroll
        for (int k = 0; k < SGEMM_V1_TILE_SIZE; ++k)
        {
            sum += A_tile[local_row][k] * B_tile[k][local_col];
        }

        // 确保所有线程使用完当前 tile 后再加载下一组数据
        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}
