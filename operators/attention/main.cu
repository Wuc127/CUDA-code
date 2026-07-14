#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "attention_v0.cuh"
#include "attention_cpu.h"
#include "check_result.h"

// 用于检查 CUDA API 是否调用成功
#define CHECK_CUDA(call)                                      \
    do                                                        \
    {                                                         \
        cudaError_t err = call;                               \
        if (err != cudaSuccess)                               \
        {                                                     \
            printf("CUDA error at %s:%d\n", __FILE__, __LINE__); \
            printf("Error: %s\n", cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)


// 初始化矩阵，随机生成 [-1, 1] 之间的 float
void random_init(float* data, int size)
{
    for (int i = 0; i < size; i++)
    {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}


int main()
{
    srand((unsigned int)time(nullptr));

    // 1. 设置 attention 的尺寸
    int bs = 2;
    int len_q = 4;
    int len_kv = 4;
    int dim = 8;

    int q_size = bs * len_q * dim;
    int k_size = bs * len_kv * dim;
    int v_size = bs * len_kv * dim;
    int s_size = bs * len_q * len_kv;
    int p_size = bs * len_q * len_kv;
    int o_size = bs * len_q * dim;

    size_t q_bytes = sizeof(float) * q_size;
    size_t k_bytes = sizeof(float) * k_size;
    size_t v_bytes = sizeof(float) * v_size;
    size_t s_bytes = sizeof(float) * s_size;
    size_t p_bytes = sizeof(float) * p_size;
    size_t o_bytes = sizeof(float) * o_size;

    // 2. 分配 CPU 内存
    float* Q_host = (float*)malloc(q_bytes);
    float* K_host = (float*)malloc(k_bytes);
    float* V_host = (float*)malloc(v_bytes);

    float* S_cpu = (float*)malloc(s_bytes);
    float* P_cpu = (float*)malloc(p_bytes);
    float* O_cpu = (float*)malloc(o_bytes);

    float* S_gpu_host = (float*)malloc(s_bytes);
    float* P_gpu_host = (float*)malloc(p_bytes);
    float* O_gpu_host = (float*)malloc(o_bytes);

    if (Q_host == nullptr || K_host == nullptr || V_host == nullptr ||
        S_cpu == nullptr || P_cpu == nullptr || O_cpu == nullptr ||
        S_gpu_host == nullptr || P_gpu_host == nullptr || O_gpu_host == nullptr)
    {
        printf("CPU malloc failed!\n");
        return 1;
    }

    // 3. 初始化 Q、K、V
    random_init(Q_host, q_size);
    random_init(K_host, k_size);
    random_init(V_host, v_size);

    // 4. 分配 GPU 内存
    float* Q_device = nullptr;
    float* K_device = nullptr;
    float* V_device = nullptr;
    float* S_device = nullptr;
    float* P_device = nullptr;
    float* O_device = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&Q_device, q_bytes));
    CHECK_CUDA(cudaMalloc((void**)&K_device, k_bytes));
    CHECK_CUDA(cudaMalloc((void**)&V_device, v_bytes));
    CHECK_CUDA(cudaMalloc((void**)&S_device, s_bytes));
    CHECK_CUDA(cudaMalloc((void**)&P_device, p_bytes));
    CHECK_CUDA(cudaMalloc((void**)&O_device, o_bytes));

    // 5. 把 Q、K、V 从 CPU 拷贝到 GPU
    CHECK_CUDA(cudaMemcpy(Q_device, Q_host, q_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(K_device, K_host, k_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(V_device, V_host, v_bytes, cudaMemcpyHostToDevice));

    // 6. 调用 GPU attention v0
    // kernel 1: S = Q @ K.T / sqrt(dim)
    {
        int threads = 256;
        int blocks = (s_size + threads - 1) / threads;
        attention_v0_score_kernel<<<blocks, threads>>>(
            Q_device,
            K_device,
            S_device,
            bs,
            len_q,
            len_kv,
            dim
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // kernel 2: P = softmax(S)
    {
        int total_rows = bs * len_q;
        int threads = 256;
        int blocks = (total_rows + threads - 1) / threads;
        attention_v0_softmax_kernel<<<blocks, threads>>>(
            S_device,
            P_device,
            bs,
            len_q,
            len_kv
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // kernel 3: O = P @ V
    {
        int threads = 256;
        int blocks = (o_size + threads - 1) / threads;
        attention_v0_output_kernel<<<blocks, threads>>>(
            P_device,
            V_device,
            O_device,
            bs,
            len_q,
            len_kv,
            dim
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // 7. 把 GPU 结果拷贝回 CPU
    CHECK_CUDA(cudaMemcpy(S_gpu_host, S_device, s_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(P_gpu_host, P_device, p_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(O_gpu_host, O_device, o_bytes, cudaMemcpyDeviceToHost));

    // 8. CPU 计算 attention
    attention_cpu(
        Q_host,
        K_host,
        V_host,
        S_cpu,
        P_cpu,
        O_cpu,
        bs,
        len_q,
        len_kv,
        dim
    );

    // 9. 检查结果
    printf("Check S result:\n");
    check_result(S_cpu, S_gpu_host, s_size);

    printf("Check P result:\n");
    check_result(P_cpu, P_gpu_host, p_size);

    printf("Check O result:\n");
    check_result(O_cpu, O_gpu_host, o_size);

    // 10. 释放 GPU 内存
    CHECK_CUDA(cudaFree(Q_device));
    CHECK_CUDA(cudaFree(K_device));
    CHECK_CUDA(cudaFree(V_device));
    CHECK_CUDA(cudaFree(S_device));
    CHECK_CUDA(cudaFree(P_device));
    CHECK_CUDA(cudaFree(O_device));

    // 11. 释放 CPU 内存
    free(Q_host);
    free(K_host);
    free(V_host);

    free(S_cpu);
    free(P_cpu);
    free(O_cpu);

    free(S_gpu_host);
    free(P_gpu_host);
    free(O_gpu_host);

    printf("Done.\n");

    return 0;
}