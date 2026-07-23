#pragma once

#include <cuda_fp16.h>

// 比较 CPU FP32 参考结果和 GPU FP16 计算结果。
//
// 返回 true 表示结果正确；
// 返回 false 表示存在超过误差范围的元素。
bool check_hgemm_result(
    const float* C_cpu,
    const __half* C_gpu,
    int M,
    int N,
    float absolute_error = 1e-2f,
    float relative_error = 1e-2f
);