#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "reduce_cpu.h"
#include "reduce_check.h"
#include "reduce_v0.cuh"
#include "reduce_v1.cuh"


// 用于检查 CUDA API 是否调用成功
#define CHECK_CUDA(call)                                         \
    do                                                           \
    {                                                            \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess)                                  \
        {                                                        \
            printf("CUDA error at %s:%d\n", __FILE__, __LINE__);  \
            printf("Error: %s\n", cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)


// 初始化数组，随机生成 [-1, 1] 之间的 float
void random_init(float* data, int size)
{
    for (int i = 0; i < size; i++)
    {
        data[i] =
            2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
    }
}


// 累加每个线程块产生的部分和
float sum_block_results(
    const float* block_results,
    int blocks
)
{
    float result = 0.0f;

    for (int i = 0; i < blocks; i++)
    {
        result += block_results[i];
    }

    return result;
}


int main()
{
    const int size = 1 << 20;

    const int threads = THREAD_PER_BLOCK;
    const int blocks = size / threads;

    const size_t input_bytes =
        size * sizeof(float);

    const size_t output_bytes =
        blocks * sizeof(float);

    // 分配 CPU 内存
    float* input_host =
        static_cast<float*>(malloc(input_bytes));

    float* output_host =
        static_cast<float*>(malloc(output_bytes));

    if (input_host == nullptr || output_host == nullptr)
    {
        printf("Failed to allocate host memory.\n");

        free(input_host);
        free(output_host);

        return 1;
    }

    // 初始化输入数据
    srand(static_cast<unsigned int>(time(nullptr)));
    random_init(input_host, size);

    // CPU 计算
    float cpu_result =
        reduce_cpu(input_host, size);

    // 分配 GPU 内存
    float* input_device = nullptr;
    float* output_device = nullptr;

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&input_device),
        input_bytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&output_device),
        output_bytes
    ));

    // =========================================================
    // Reduce v0：使用 Global Memory
    // =========================================================

    // v0 会修改 input_device，因此运行前重新复制输入数据
    CHECK_CUDA(cudaMemcpy(
        input_device,
        input_host,
        input_bytes,
        cudaMemcpyHostToDevice
    ));

    reduce_v0_kernel<<<blocks, threads>>>(
        input_device,
        output_device
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(
        output_host,
        output_device,
        output_bytes,
        cudaMemcpyDeviceToHost
    ));

    float gpu_result_v0 =
        sum_block_results(output_host, blocks);

    printf("Reduce v0:\n");

    reduce_check(
        &cpu_result,
        &gpu_result_v0
    );

    printf("\n");

    // =========================================================
    // Reduce v1：使用 Shared Memory
    // =========================================================

    // 重新复制输入，保证不同版本使用相同的原始数据
    CHECK_CUDA(cudaMemcpy(
        input_device,
        input_host,
        input_bytes,
        cudaMemcpyHostToDevice
    ));

    reduce_v1_kernel<<<blocks, threads>>>(
        input_device,
        output_device
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(
        output_host,
        output_device,
        output_bytes,
        cudaMemcpyDeviceToHost
    ));

    float gpu_result_v1 =
        sum_block_results(output_host, blocks);

    printf("Reduce v1:\n");

    reduce_check(
        &cpu_result,
        &gpu_result_v1
    );

    // 释放 GPU 内存
    CHECK_CUDA(cudaFree(input_device));
    CHECK_CUDA(cudaFree(output_device));

    // 释放 CPU 内存
    free(input_host);
    free(output_host);

    return 0;
}

