#pragma once

#include <cuda_runtime.h>

__global__ void attention_v0_score_kernel(
    const float* Q,
    const float* K,
    float* S,
    int bs,
    int len_q,
    int len_kv,
    int dim
);

__global__ void attention_v0_softmax_kernel(
    const float* S,
    float* P,
    int bs,
    int len_q,
    int len_kv
);

__global__ void attention_v0_output_kernel(
    const float* P,
    const float* V,
    float* O,
    int bs,
    int len_q,
    int len_kv,
    int dim
);