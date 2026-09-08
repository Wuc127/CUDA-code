#include "transpose_v0.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// 每个线程负责转置一个元素：Y[col][row] = X[row][col]
__global__ void transpose_v0_kernel(
    const float* X,
    float* Y,
    int M,
    int N
)
{
    // 当前线程负责的输入矩阵行号和列号
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 防止线程访问矩阵范围之外的数据
    if (row < M && col < N)
    {
        // 输入为 M × N，输出为 N × M，均按行优先存储
        Y[col * M + row] = X[row * N + col];
    }
}
