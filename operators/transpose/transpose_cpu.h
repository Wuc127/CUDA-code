#pragma once


// CPU 上计算矩阵转置，输入和输出均按行优先存储
// X: M × N
// Y: N × M，使用与 X 独立的存储空间
void transpose_cpu(
    const float* X,
    float* Y,
    int M,
    int N
);
