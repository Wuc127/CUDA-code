#include "reduce_v3.cuh"
#include <device_launch_parameters.h>


__global__ void reduce_v3_kernel(float *d_input, float *d_output)
{
    __shared__ float shared[THREAD_PER_BLOCK];
    float *input_begin = d_input + blockDim.x * blockIdx.x;
    shared[threadIdx.x] = input_begin[threadIdx.x];
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i)
        {
            shared[threadIdx.x] += shared[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        d_output[blockIdx.x] = shared[0];
}