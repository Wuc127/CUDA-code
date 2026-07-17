#include "elementwise_v3.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// 使用 Grid-Stride Loop 和 float4 向量化访存完成逐元素加法
__global__ void elementwise_v3_kernel(
    const float* A,
    const float* B,
    float* C,
    int num_elements
)
{
    // 当前线程在线性网格中的全局编号
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 整个网格中的线程总数
    int stride = blockDim.x * gridDim.x;

    // 完整的 float4 数量
    int vec_count = num_elements / 4;

    const float4* A_vec = reinterpret_cast<const float4*>(A);
    const float4* B_vec = reinterpret_cast<const float4*>(B);
    float4* C_vec = reinterpret_cast<float4*>(C);

    // 每个线程以整个网格的线程总数为步长处理多个 float4
    for (int vec_idx = thread_idx; vec_idx < vec_count; vec_idx += stride)
    {
        // 一次读取连续的 4 个 float
        float4 a = A_vec[vec_idx];
        float4 b = B_vec[vec_idx];

        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;

        // 一次写回连续的 4 个 float
        C_vec[vec_idx] = c;
    }

    // 处理 num_elements 不能被 4 整除时剩余的元素
    int remaining_start = vec_count * 4;
    int remaining_idx = remaining_start + thread_idx;

    if (remaining_idx < num_elements)
    {
        C[remaining_idx] = A[remaining_idx] + B[remaining_idx];
    }
}