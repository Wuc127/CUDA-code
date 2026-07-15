#include "elementwise_v0.cuh"
#include <cuda_runtime.h>


// 每个线程负责计算一个元素：C[i] = A[i] + B[i]
__global__ void elementwise_v0_kernel(
    const float* A,
    const float* B,
    float* C,
    int num_elements
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements)
    {
        C[idx] = A[idx] + B[idx];
    }
}