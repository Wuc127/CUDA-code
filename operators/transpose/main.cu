#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "transpose_cpu.h"
#include "transpose_v0.cuh"
#include "transpose_v1.cuh"
#include "transpose_v2.cuh"
#include "transpose_v3.cuh"
#include "transpose_v4.cuh"
#include "transpose_check.h"


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

    // 使用非方阵，且尺寸不是 4、32、64 的整数倍，测试分块和向量化的边界处理
    int M = 65;
    int N = 99;
    int num_elements = M * N;
    size_t bytes = sizeof(float) * num_elements;

    // 分配 CPU 内存，输入为 M × N，输出为 N × M
    float* X_host = (float*)malloc(bytes);
    float* Y_cpu = (float*)malloc(bytes);
    float* Y_gpu_host = (float*)malloc(bytes);

    if (X_host == nullptr ||
        Y_cpu == nullptr ||
        Y_gpu_host == nullptr)
    {
        printf("CPU malloc failed!\n");
        free(X_host);
        free(Y_cpu);
        free(Y_gpu_host);
        return 1;
    }

    // 初始化输入矩阵
    random_init(X_host, num_elements);

    // 在 CPU 上计算正确结果
    transpose_cpu(
        X_host,
        Y_cpu,
        M,
        N
    );

    // 分配 GPU 内存，输入和输出使用独立的存储空间
    float* X_device = nullptr;
    float* Y_device = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&X_device, bytes));
    CHECK_CUDA(cudaMalloc((void**)&Y_device, bytes));

    // 把输入矩阵从 CPU 拷贝到 GPU
    CHECK_CUDA(cudaMemcpy(
        X_device,
        X_host,
        bytes,
        cudaMemcpyHostToDevice
    ));

    printf("Transpose: %d x %d -> %d x %d\n", M, N, N, M);

    // 调用 GPU transpose v0
    {
        // 每个线程负责一个元素，x 方向对应输入矩阵的列
        dim3 threads(16, 16);
        dim3 blocks(
            (N + threads.x - 1) / threads.x,
            (M + threads.y - 1) / threads.y
        );

        // 将输出填为 NaN，避免上一个版本的结果掩盖当前版本漏写的元素
        CHECK_CUDA(cudaMemset(Y_device, 0xFF, bytes));

        transpose_v0_kernel<<<blocks, threads>>>(
            X_device,
            Y_device,
            M,
            N
        );

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(
            Y_gpu_host,
            Y_device,
            bytes,
            cudaMemcpyDeviceToHost
        ));

        printf("Check transpose v0 result:\n");

        check_transpose_result(
            Y_cpu,
            Y_gpu_host,
            num_elements
        );
    }

    // 调用 GPU transpose v1
    {
        // 一个线程块搬运一个 32 × 32 Tile
        const int TILE_DIM = TRANSPOSE_V1_TILE_DIM;
        dim3 threads(TILE_DIM, TILE_DIM);
        dim3 blocks(
            (N + TILE_DIM - 1) / TILE_DIM,
            (M + TILE_DIM - 1) / TILE_DIM
        );

        // 将输出填为 NaN，避免上一个版本的结果掩盖当前版本漏写的元素
        CHECK_CUDA(cudaMemset(Y_device, 0xFF, bytes));

        transpose_v1_kernel<<<blocks, threads>>>(
            X_device,
            Y_device,
            M,
            N
        );

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(
            Y_gpu_host,
            Y_device,
            bytes,
            cudaMemcpyDeviceToHost
        ));

        printf("Check transpose v1 result:\n");

        check_transpose_result(
            Y_cpu,
            Y_gpu_host,
            num_elements
        );
    }

    // 调用 GPU transpose v2
    {
        // 线程映射与 v1 相同，Kernel 内部增加 Padding
        const int TILE_DIM = TRANSPOSE_V2_TILE_DIM;
        dim3 threads(TILE_DIM, TILE_DIM);
        dim3 blocks(
            (N + TILE_DIM - 1) / TILE_DIM,
            (M + TILE_DIM - 1) / TILE_DIM
        );

        // 将输出填为 NaN，避免上一个版本的结果掩盖当前版本漏写的元素
        CHECK_CUDA(cudaMemset(Y_device, 0xFF, bytes));

        transpose_v2_kernel<<<blocks, threads>>>(
            X_device,
            Y_device,
            M,
            N
        );

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(
            Y_gpu_host,
            Y_device,
            bytes,
            cudaMemcpyDeviceToHost
        ));

        printf("Check transpose v2 result:\n");

        check_transpose_result(
            Y_cpu,
            Y_gpu_host,
            num_elements
        );
    }

    // 调用 GPU transpose v3
    {
        // 每个线程搬运连续的 4 个元素，一个线程块处理 32 × 32 Tile
        const int TILE_DIM = TRANSPOSE_V3_TILE_DIM;
        dim3 threads(TILE_DIM / 4, TILE_DIM);
        dim3 blocks(
            (N + TILE_DIM - 1) / TILE_DIM,
            (M + TILE_DIM - 1) / TILE_DIM
        );

        // 将输出填为 NaN，避免上一个版本的结果掩盖当前版本漏写的元素
        CHECK_CUDA(cudaMemset(Y_device, 0xFF, bytes));

        transpose_v3_kernel<<<blocks, threads>>>(
            X_device,
            Y_device,
            M,
            N
        );

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(
            Y_gpu_host,
            Y_device,
            bytes,
            cudaMemcpyDeviceToHost
        ));

        printf("Check transpose v3 result:\n");

        check_transpose_result(
            Y_cpu,
            Y_gpu_host,
            num_elements
        );
    }

    // 调用 GPU transpose v4
    {
        // 128 个线程通过循环搬运一个 64 × 64 Tile
        const int TILE_DIM = TRANSPOSE_V4_TILE_DIM;
        dim3 threads(TILE_DIM / 4, 8);
        dim3 blocks(
            (N + TILE_DIM - 1) / TILE_DIM,
            (M + TILE_DIM - 1) / TILE_DIM
        );

        // 将输出填为 NaN，避免上一个版本的结果掩盖当前版本漏写的元素
        CHECK_CUDA(cudaMemset(Y_device, 0xFF, bytes));

        transpose_v4_kernel<<<blocks, threads>>>(
            X_device,
            Y_device,
            M,
            N
        );

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(
            Y_gpu_host,
            Y_device,
            bytes,
            cudaMemcpyDeviceToHost
        ));

        printf("Check transpose v4 result:\n");

        check_transpose_result(
            Y_cpu,
            Y_gpu_host,
            num_elements
        );
    }

    // 释放 GPU 内存
    CHECK_CUDA(cudaFree(X_device));
    CHECK_CUDA(cudaFree(Y_device));

    // 释放 CPU 内存
    free(X_host);
    free(Y_cpu);
    free(Y_gpu_host);

    printf("Done.\n");

    return 0;
}
