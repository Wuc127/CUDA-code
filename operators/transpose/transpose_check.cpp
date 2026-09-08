#include "transpose_check.h"

#include <cmath>
#include <cstdio>


void check_transpose_result(
    const float* Y_cpu,
    const float* Y_gpu,
    int size
)
{
    const float tolerance = 1e-5f;

    for (int i = 0; i < size; i++)
    {
        float error = fabsf(Y_cpu[i] - Y_gpu[i]);

        // 同时避免 NaN 误差被误判为校验通过
        if (!(error <= tolerance))
        {
            printf("Result mismatch at index %d\n", i);
            printf("CPU result: %f\n", Y_cpu[i]);
            printf("GPU result: %f\n", Y_gpu[i]);
            printf("Error: %f\n", error);
            return;
        }
    }

    printf("Result is correct.\n");
}
