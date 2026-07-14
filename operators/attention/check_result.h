#pragma once

bool check_result(
    const float* cpu_out,
    const float* gpu_out,
    int size,
    float eps = 1e-4f
);