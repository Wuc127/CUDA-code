#pragma once


// CPU 上计算 C = A × B
// A: M × K
// B: K × N
// C: M × N
void sgemm_cpu(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);