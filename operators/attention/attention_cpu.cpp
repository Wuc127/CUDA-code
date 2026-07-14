#include "attention_cpu.h"

#include <cmath>    // sqrtf, expf
#include <cfloat>   // FLT_MAX

// CPU 版本 attention:
// S = Q @ K.T / sqrt(dim)
// P = softmax(S)
// O = P @ V
// 注意：S、P、O 都由 main 函数提前分配好
void attention_cpu(
    const float* Q,   // Q: [bs, len_q, dim]
    const float* K,   // K: [bs, len_kv, dim]
    const float* V,   // V: [bs, len_kv, dim]
    float* S,         // S: [bs, len_q, len_kv]
    float* P,         // P: [bs, len_q, len_kv]
    float* O,         // O: [bs, len_q, dim]
    int bs,
    int len_q,
    int len_kv,
    int dim
)
{
    float scale = 1.0f / sqrtf((float)dim);

    // 第一步：计算 S = Q @ K.T / sqrt(dim)
    for (int b = 0; b < bs; b++)
    {
        for (int q = 0; q < len_q; q++)
        {
            for (int kv = 0; kv < len_kv; kv++)
            {
                float sum = 0.0f;

                for (int d = 0; d < dim; d++)
                {
                    float q_val = Q[(b * len_q + q) * dim + d];      // Q[b, q, d]
                    float k_val = K[(b * len_kv + kv) * dim + d];    // K[b, kv, d]

                    sum += q_val * k_val;
                }

                S[(b * len_q + q) * len_kv + kv] = sum * scale;      // S[b, q, kv]
            }
        }
    }

    // 第二步：对 S 的最后一维做 softmax，得到 P
    // P[b, q, :] = softmax(S[b, q, :])
    for (int b = 0; b < bs; b++)
    {
        for (int q = 0; q < len_q; q++)
        {
            int row_start = (b * len_q + q) * len_kv;

            // 1. 找当前行最大值，保证 softmax 数值稳定
            float row_max = -FLT_MAX;

            for (int kv = 0; kv < len_kv; kv++)
            {
                float val = S[row_start + kv];

                if (val > row_max)
                {
                    row_max = val;
                }
            }

            // 2. 计算 exp(S - row_max)，同时求和
            float row_sum = 0.0f;

            for (int kv = 0; kv < len_kv; kv++)
            {
                float exp_val = expf(S[row_start + kv] - row_max);

                P[row_start + kv] = exp_val;
                row_sum += exp_val;
            }

            // 3. 归一化
            for (int kv = 0; kv < len_kv; kv++)
            {
                P[row_start + kv] = P[row_start + kv] / row_sum;
            }
        }
    }

    // 第三步：计算 O = P @ V
    // O[b, q, d] = sum_kv P[b, q, kv] * V[b, kv, d]
    for (int b = 0; b < bs; b++)
    {
        for (int q = 0; q < len_q; q++)
        {
            for (int d = 0; d < dim; d++)
            {
                float sum = 0.0f;

                for (int kv = 0; kv < len_kv; kv++)
                {
                    float p_val = P[(b * len_q + q) * len_kv + kv];  // P[b, q, kv]
                    float v_val = V[(b * len_kv + kv) * dim + d];    // V[b, kv, d]

                    sum += p_val * v_val;
                }

                O[(b * len_q + q) * dim + d] = sum;                  // O[b, q, d]
            }
        }
    }
}