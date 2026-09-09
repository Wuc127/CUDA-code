#pragma once

#include <cublas_v2.h>


// 使用 cuBLAS 计算行主序矩阵乘法：
// C = alpha * A * B + beta * C
// A: M x K
// B: K x N
// C: M x N
cublasStatus_t sgemm_cublas(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha = 1.0f,
    float beta = 0.0f
);
