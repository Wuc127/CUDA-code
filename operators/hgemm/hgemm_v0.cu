#include "hgemm_v0.cuh"
#include <device_launch_parameters.h>


__global__ void hgemm_v0_kernel(
    const __half* A,
    const __half* B,
    __half* C,
    int M,
    int N,
    int K
)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < K; k++)
        {
            float a = __half2float(A[row * K + k]);
            float b = __half2float(B[k * N + col]);

            sum += a * b;
        }

        C[row * N + col] = __float2half(sum);
    }
}