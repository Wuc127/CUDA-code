#include "hgemm_cpu.h"


void hgemm_cpu(
    const __half* A,
    const __half* B,
    float* C,
    int M,
    int N,
    int K
)
{
    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < N; col++)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; k++)
            {
                float a = __half2float(A[row * K + k]);
                float b = __half2float(B[k * N + col]);

                sum += a * b;
            }

            C[row * N + col] = sum;
        }
    }
}