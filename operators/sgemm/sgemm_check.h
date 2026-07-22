#pragma once


bool check_sgemm_result(
    const float* C_cpu,
    const float* C_gpu,
    int M,
    int N
);