#include "sgemm_v0.cuh"


// Naive SGEMM：一个线程计算 C 中的一个元素
__global__ void sgemm_v0_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{
    // 当前线程负责计算的 C 矩阵行号和列号
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 防止线程访问矩阵范围之外的数据
    if (row < M && col < N)
    {
        float sum = 0.0f;

        // 计算 C[row][col]
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}