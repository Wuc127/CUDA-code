#pragma once

#include <cuda_fp16.h>


// 朴素 HGEMM：C = A × B
//
// A: [M, K]，行主序
// B: [K, N]，行主序
// C: [M, N]，行主序
//
// 一个线程计算 C 中的一个元素。
__global__ void hgemm_v0_kernel(
    const __half* A,
    const __half* B,
    __half* C,
    int M,
    int N,
    int K
);