#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "elementwise_v0.cuh"
#include "elementwise_check.h"


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
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}


int main()
{
    srand((unsigned int)time(nullptr));

    // 1. 设置数组大小
    int num_elements = 1024;

    size_t bytes = sizeof(float) * num_elements;

    // 2. 分配 CPU 内存
    float* A_host = (float*)malloc(bytes);
    float* B_host = (float*)malloc(bytes);

    float* C_cpu = (float*)malloc(bytes);
    float* C_gpu_host = (float*)malloc(bytes);

    if (A_host == nullptr || B_host == nullptr ||
        C_cpu == nullptr || C_gpu_host == nullptr)
    {
        printf("CPU malloc failed!\n");
        return 1;
    }

    // 3. 初始化输入数组 A 和 B
    random_init(A_host, num_elements);
    random_init(B_host, num_elements);

    // 4. 在 CPU 上计算正确结果
    for (int i = 0; i < num_elements; i++)
    {
        C_cpu[i] = A_host[i] + B_host[i];
    }

    // 5. 分配 GPU 内存
    float* A_device = nullptr;
    float* B_device = nullptr;
    float* C_device = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&A_device, bytes));
    CHECK_CUDA(cudaMalloc((void**)&B_device, bytes));
    CHECK_CUDA(cudaMalloc((void**)&C_device, bytes));

    // 6. 把输入数组从 CPU 拷贝到 GPU
    CHECK_CUDA(cudaMemcpy(
        A_device,
        A_host,
        bytes,
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        B_device,
        B_host,
        bytes,
        cudaMemcpyHostToDevice
    ));

    // 7. 调用 GPU elementwise v0
    {
        int threads = 256;
        int blocks = (num_elements + threads - 1) / threads;

        elementwise_v0_kernel<<<blocks, threads>>>(
            A_device,
            B_device,
            C_device,
            num_elements
        );

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // 8. 把 GPU 结果拷贝回 CPU
    CHECK_CUDA(cudaMemcpy(
        C_gpu_host,
        C_device,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // 9. 检查结果
    printf("Check elementwise v0 result:\n");
    check_elementwise_result(
        C_cpu,
        C_gpu_host,
        num_elements
    );

    // 10. 释放 GPU 内存
    CHECK_CUDA(cudaFree(A_device));
    CHECK_CUDA(cudaFree(B_device));
    CHECK_CUDA(cudaFree(C_device));

    // 11. 释放 CPU 内存
    free(A_host);
    free(B_host);
    free(C_cpu);
    free(C_gpu_host);

    printf("Done.\n");

    return 0;
}