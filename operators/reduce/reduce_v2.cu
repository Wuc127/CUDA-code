#include "reduce_v2.cuh"
#include <device_launch_parameters.h>


__global__ void reduce_v2_kernel(float *d_input, float *d_output)
{
    __shared__ float shared[THREAD_PER_BLOCK];
    float *input_begin = d_input + blockDim.x * blockIdx.x;
    shared[threadIdx.x] = input_begin[threadIdx.x];
    __syncthreads();

    for (int i = 1; i < blockDim.x; i *= 2)
    {
        if (threadIdx.x < blockDim.x / (2 * i))
        {
            int index = threadIdx.x * 2 * i;
            shared[index] += shared[index + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        d_output[blockIdx.x] = shared[0];
}