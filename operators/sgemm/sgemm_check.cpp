#include "sgemm_check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>


bool check_sgemm_result(
    const float* C_cpu,
    const float* C_gpu,
    int M,
    int N
)
{
    constexpr float absolute_tolerance = 1e-4f;
    constexpr float relative_tolerance = 1e-3f;

    float max_absolute_error = 0.0f;
    float max_relative_error = 0.0f;

    const int num_elements = M * N;

    for (int index = 0; index < num_elements; ++index)
    {
        const float expected = C_cpu[index];
        const float actual = C_gpu[index];

        const float absolute_error =
            std::fabs(actual - expected);

        const float relative_error =
            absolute_error /
            std::max(std::fabs(expected), 1e-6f);

        max_absolute_error =
            std::max(max_absolute_error, absolute_error);

        max_relative_error =
            std::max(max_relative_error, relative_error);

        const float allowed_error =
            absolute_tolerance +
            relative_tolerance * std::fabs(expected);

        if (absolute_error > allowed_error)
        {
            const int row = index / N;
            const int col = index % N;

            std::printf("SGEMM result check failed.\n");
            std::printf(
                "Position: row = %d, col = %d\n",
                row,
                col
            );
            std::printf(
                "CPU result: %.8f\n",
                expected
            );
            std::printf(
                "GPU result: %.8f\n",
                actual
            );
            std::printf(
                "Absolute error: %.8f\n",
                absolute_error
            );
            std::printf(
                "Allowed error:  %.8f\n",
                allowed_error
            );

            return false;
        }
    }

    std::printf("SGEMM result check passed.\n");
    std::printf(
        "Maximum absolute error: %.8f\n",
        max_absolute_error
    );
    std::printf(
        "Maximum relative error: %.8f\n",
        max_relative_error
    );

    return true;
}