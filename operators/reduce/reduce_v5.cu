#include "reduce_v5.cuh"
#include <device_launch_parameters.h>



__device__ void warpReduce(volatile float *cache, unsigned int tid)
{
    cache[tid] += cache[tid + 32];
    //__syncthreads();
    cache[tid] += cache[tid + 16];
    //__syncthreads();
    cache[tid] += cache[tid + 8];
    //__syncthreads();
    cache[tid] += cache[tid + 4];
    //__syncthreads();
    cache[tid] += cache[tid + 2];
    //__syncthreads();
    cache[tid] += cache[tid + 1];
    //__syncthreads();
}

__global__ void reduce_v5_kernel(float *d_input, float *d_output)
{
    int tid = threadIdx.x;
    __shared__ float shared[THREAD_PER_BLOCK];
    float *input_begin = d_input + blockDim.x * blockIdx.x * 2;
    shared[tid] = input_begin[tid] + input_begin[tid + blockDim.x];
    __syncthreads();

    for (int i = blockDim.x / 2; i > 32; i /= 2)
    {
        if (tid < i)
            shared[tid] += shared[tid + i];
        __syncthreads();
    }
    if (tid < 32)
    {
        warpReduce(shared, tid);
    }

    if (tid == 0)
        d_output[blockIdx.x] = shared[0];
}