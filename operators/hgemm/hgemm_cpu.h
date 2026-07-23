#pragma once

#include <cuda_fp16.h>

// CPU 参考实现：C = A × B
// A: [M, K]
// B: [K, N]
// C: [M, N]
//
// A、B 使用 FP16 存储，C 使用 FP32 保存参考结果。
void hgemm_cpu(
    const __half* A,
    const __half* B,
    float* C,
    int M,
    int N,
    int K
);