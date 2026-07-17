#include "elementwise_v2.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// 使用 float4 向量化访存完成逐元素加法：C[i] = A[i] + B[i]
// 每个线程负责连续的 4 个 float 元素
__global__ void elementwise_v2_kernel(
    const float* A,
    const float* B,
    float* C,
    int num_elements
)
{
    // 当前线程负责的 float4 编号
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 当前线程负责的第一个 float 元素下标
    int idx = vec_idx * 4;

    // 当前线程负责完整的 4 个元素
    if (idx + 3 < num_elements)
    {
        const float4* A_vec = reinterpret_cast<const float4*>(A);
        const float4* B_vec = reinterpret_cast<const float4*>(B);
        float4* C_vec = reinterpret_cast<float4*>(C);

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
    else
    {
        // 处理 num_elements 不能被 4 整除时剩余的元素
        for (int i = idx; i < num_elements; i++)
        {
            C[i] = A[i] + B[i];
        }
    }
}