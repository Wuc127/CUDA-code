#include "transpose_cpu.h"

#include <cstddef>


// CPU 上计算矩阵转置：Y[col][row] = X[row][col]
void transpose_cpu(
    const float* X,
    float* Y,
    int M,
    int N
)
{
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            Y[static_cast<std::size_t>(col) * M + row] =
                X[static_cast<std::size_t>(row) * N + col];
        }
    }
}
