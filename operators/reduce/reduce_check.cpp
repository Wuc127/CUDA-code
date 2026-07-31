#include "reduce_check.h"

#include <cmath>
#include <cstdio>


bool reduce_check(const float* cpu_result,
                  const float* gpu_result,
                  float eps)
{
    float error = std::fabs(*cpu_result - *gpu_result);

    printf("CPU result: %.6f\n", *cpu_result);
    printf("GPU result: %.6f\n", *gpu_result);
    printf("Error:      %.6e\n", error);

    if (error <= eps)
    {
        printf("Reduce result is correct.\n");
        return true;
    }

    printf("Reduce result is incorrect.\n");
    return false;
}