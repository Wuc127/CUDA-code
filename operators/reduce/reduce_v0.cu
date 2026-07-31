#include "reduce_v0.cuh"
#include <device_launch_parameters.h>


__global__ void reduce_v0_kernel(float* input, float* output)
{
    float* input_begin = input + blockIdx.x * blockDim.x;

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (threadIdx.x % (2 * stride) == 0)
        {
            input_begin[threadIdx.x] += input_begin[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = input_begin[0];
    }
}