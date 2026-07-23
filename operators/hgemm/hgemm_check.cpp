#include "hgemm_check.h"

#include <cmath>
#include <cstdio>


bool check_hgemm_result(
    const float* C_cpu,
    const __half* C_gpu,
    int M,
    int N,
    float absolute_error,
    float relative_error
)
{
    int num_elements = M * N;
    int error_count = 0;

    float max_absolute_error = 0.0f;
    float max_relative_error = 0.0f;

    int max_error_index = -1;

    for (int i = 0; i < num_elements; i++)
    {
        float cpu_value = C_cpu[i];
        float gpu_value = __half2float(C_gpu[i]);

        float abs_error = std::fabs(cpu_value - gpu_value);

        float denominator = std::fmax(std::fabs(cpu_value), 1e-6f);
        float rel_error = abs_error / denominator;

        if (abs_error > max_absolute_error)
        {
            max_absolute_error = abs_error;
            max_error_index = i;
        }

        if (rel_error > max_relative_error)
        {
            max_relative_error = rel_error;
        }

        // 绝对误差和相对误差都超过阈值时，才认为结果错误。
        if (abs_error > absolute_error &&
            rel_error > relative_error)
        {
            if (error_count < 10)
            {
                int row = i / N;
                int col = i % N;

                std::printf(
                    "Mismatch at C[%d][%d]: "
                    "CPU = %.6f, GPU = %.6f, "
                    "absolute error = %.6f, relative error = %.6f\n",
                    row,
                    col,
                    cpu_value,
                    gpu_value,
                    abs_error,
                    rel_error
                );
            }

            error_count++;
        }
    }

    if (error_count == 0)
    {
        std::printf("Result check passed.\n");
        std::printf(
            "Maximum absolute error: %.6f\n",
            max_absolute_error
        );
        std::printf(
            "Maximum relative error: %.6f\n",
            max_relative_error
        );

        return true;
    }

    std::printf("Result check failed.\n");
    std::printf(
        "Incorrect elements: %d / %d\n",
        error_count,
        num_elements
    );
    std::printf(
        "Maximum absolute error: %.6f\n",
        max_absolute_error
    );
    std::printf(
        "Maximum relative error: %.6f\n",
        max_relative_error
    );

    if (max_error_index >= 0)
    {
        int row = max_error_index / N;
        int col = max_error_index % N;

        std::printf(
            "Maximum absolute error at C[%d][%d].\n",
            row,
            col
        );
    }

    return false;
}