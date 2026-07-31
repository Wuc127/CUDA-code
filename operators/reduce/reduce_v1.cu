#include "reduce_v1.cuh"
#include <device_launch_parameters.h>


__global__ void reduce_v1_kernel(
    const float* input,
    float* output
)
{
    __shared__ float shared[THREAD_PER_BLOCK];

    int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    // 将全局内存数据加载到共享内存
    shared[threadIdx.x] = input[index];

    __syncthreads();

    // 在共享内存中进行归约
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (threadIdx.x % (2 * stride) == 0)
        {
            shared[threadIdx.x] +=
                shared[threadIdx.x + stride];
        }

        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = shared[0];
    }
}
