#include "elementwise_cpu.h"

void elementwise_cpu(
    const float* A,
    const float* B,
    float* C,
    int num_elements
)
{
    for (int i = 0; i < num_elements; i++)
    {
        C[i] = A[i] + B[i];
    }
}