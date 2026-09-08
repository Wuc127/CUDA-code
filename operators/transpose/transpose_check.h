#pragma once


// Y_cpu、Y_gpu 均为主机端的转置结果，size 为元素总数 M × N
void check_transpose_result(
    const float* Y_cpu,
    const float* Y_gpu,
    int size
);
