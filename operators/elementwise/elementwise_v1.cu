#include "elementwise_v1.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// 使用 Grid-Stride Loop 完成逐元素加法：C[i] = A[i] + B[i]
__global__ void elementwise_v1_kernel(
    const float* A,
    const float* B,
    float* C,
    int num_elements
)
{
    // 当前线程在线性网格中的全局编号
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 整个网格中的线程总数
    int stride = blockDim.x * gridDim.x;

    // 每个线程以整个网格的线程总数为步长处理多个元素
    for (int i = idx; i < num_elements; i += stride)
    {
        C[i] = A[i] + B[i];
    }
}