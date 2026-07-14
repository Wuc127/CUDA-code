#include "check_result.h"

#include <cstdio>
#include <cmath>

// 比较 CPU 和 GPU 的输出结果是否一致
bool check_result(
    const float* cpu_out,
    const float* gpu_out,
    int size,
    float eps
)
{
    for (int i = 0; i < size; i++)
    {
        float diff = fabsf(cpu_out[i] - gpu_out[i]);

        if (diff > eps)
        {
            printf("Result mismatch at index %d\n", i);
            printf("CPU result = %.8f\n", cpu_out[i]);
            printf("GPU result = %.8f\n", gpu_out[i]);
            printf("Diff       = %.8f\n", diff);

            return false;
        }
    }

    printf("Check passed! CPU and GPU results match.\n");
    return true;
}