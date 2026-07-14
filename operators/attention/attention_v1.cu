#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// kernel 1 v1: S = Q @ K.T / sqrt(dim)
// 优化点：使用二维 block + 三维 grid，直接映射到 S[b, q, kv]
__global__ void attention_v1_score_kernel(
    const float* Q,   // Q: [bs, len_q, dim]      查询矩阵
    const float* K,   // K: [bs, len_kv, dim]     键矩阵
    float* S,         // S: [bs, len_q, len_kv]   注意力分数矩阵（输出）
    int bs,
    int len_q,
    int len_kv,
    int dim
)
{
    // threadIdx.x / blockIdx.x 负责 kv 方向
    // threadIdx.y / blockIdx.y 负责 q 方向
    // blockIdx.z 负责 batch 方向
    int kv = blockIdx.x * blockDim.x + threadIdx.x;  // 第几个 key 位置
    int q  = blockIdx.y * blockDim.y + threadIdx.y;  // 第几个 query 位置
    int b  = blockIdx.z;                             // 第几个 batch

    // 边界判断：
    // 因为 len_q 和 len_kv 不一定刚好能被 blockDim.y / blockDim.x 整除，
    // 所以多出来的线程需要直接返回。
    if (b >= bs || q >= len_q || kv >= len_kv)
    {
        return;
    }

    // 每个线程仍然负责计算一个 S[b, q, kv]
    // 即 Q[b, q, :] 和 K[b, kv, :] 的点积
    float sum = 0.0f;
    for (int d = 0; d < dim; d++)
    {
        float q_val = Q[(b * len_q + q) * dim + d];      // Q[b, q, d]
        float k_val = K[(b * len_kv + kv) * dim + d];    // K[b, kv, d]
        sum += q_val * k_val;
    }

    float scale = rsqrtf((float)dim);

    // 写入 S[b, q, kv]
    S[(b * len_q + q) * len_kv + kv] = sum * scale;
}


// P = softmax(S)  对 S 的每一行做 softmax
__global__ void attention_v0_softmax_kernel(
    const float* S,  // [bs, len_q, len_kv]  注意力分数矩阵（输入）
    float* P,        // [bs, len_q, len_kv]  注意力概率矩阵（输出）
    int bs,
    int len_q,
    int len_kv
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = bs * len_q;   // 总共需要做 softmax 的行数 = bs * len_q
    if (idx >= total_rows)
    {
        return;
    }

    int q = idx % len_q;    // 第几个 query 位置
    int b = idx / len_q;    // 第几个 batch
    int row_start = (b * len_q + q) * len_kv;    // 当前行的起始偏移量, S[b, q, 0] 在一维数组中的位置

    // 第一步：找这一行的最大值，softmax 直接算 exp(S) 可能会数值溢出，
    // 所以通常写成：softmax(x_i) = exp(x_i - max) / sum_j exp(x_j - max)
    float row_max = -FLT_MAX;
    for (int kv = 0; kv < len_kv; kv++)
    {
        float val = S[row_start + kv];
        if (val > row_max)
        {
            row_max = val;
        }
    }

    // 第二步：计算 exp(S - row_max)，同时求和
    float row_sum = 0.0f;
    for (int kv = 0; kv < len_kv; kv++)
    {
        float exp_val = expf(S[row_start + kv] - row_max);
        P[row_start + kv] = exp_val;
        row_sum += exp_val;
    }

    // 第三步：归一化
    for (int kv = 0; kv < len_kv; kv++)
    {
        P[row_start + kv] = P[row_start + kv] / row_sum;
    }
}



// kernel 3: O = P @ V
__global__ void attention_v0_output_kernel(
    const float* P,   // [bs, len_q, len_kv]  注意力分数矩阵 (Softmax 输出)
    const float* V,   // [bs, len_kv, dim]    值矩阵
    float* O,         // [bs, len_q, dim]     输出矩阵
    int bs,
    int len_q,
    int len_kv,
    int dim
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bs * len_q * dim;  // 总共需要计算 O 的元素个数, 每个线程负责计算一个输出元素
    if (idx >= total)
    {
        return;
    }

    // 从一维索引 idx 解码出三维坐标 (b, q, d)  O 的形状是 [bs, len_q, dim]
    int d = idx % dim;              // 第几个维度（变化最快）
    int q = (idx / dim) % len_q;    // 第几个 query 位置
    int b = idx / (len_q * dim);    // 第几个 batch（变化最慢）

    float sum = 0.0f;
    for (int kv = 0; kv < len_kv; kv++)
    {
        float p_val = P[(b * len_q + q) * len_kv + kv]; // P[b, q, kv]（当前 query 对第 kv 个 key 的注意力权重）
        float v_val = V[(b * len_kv + kv) * dim + d];   // V[b, kv, d]（第 kv 个 key 的第 d 个维度值）
        sum += p_val * v_val;
    }
    O[(b * len_q + q) * dim + d] = sum;    // 写入 O[b, q, d]
    
}
