#include "elementwise_check.h"

#include <cmath>
#include <cstdio>


void check_elementwise_result(
    const float* C_cpu,
    const float* C_gpu,
    int size
)
{
    const float tolerance = 1e-5f;

    for (int i = 0; i < size; i++)
    {
        float error = fabsf(C_cpu[i] - C_gpu[i]);

        if (error > tolerance)
        {
            printf("Result mismatch at index %d\n", i);
            printf("CPU result: %f\n", C_cpu[i]);
            printf("GPU result: %f\n", C_gpu[i]);
            printf("Error: %f\n", error);
            return;
        }
    }

    printf("Result is correct.\n");
}