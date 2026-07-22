#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "sgemm_v0.cuh"
#include "sgemm_cpu.h"
#include "sgemm_check.h"


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
void random_init(float* data, int num_elements)
{
    for (int i = 0; i < num_elements; ++i)
    {
        data[i] =
            2.0f * static_cast<float>(rand()) /
            static_cast<float>(RAND_MAX) - 1.0f;
    }
}


int main()
{
    // A: M × K
    // B: K × N
    // C: M × N
    const int M = 512;
    const int N = 512;
    const int K = 512;

    const int A_num_elements = M * K;
    const int B_num_elements = K * N;
    const int C_num_elements = M * N;

    const size_t A_bytes =
        static_cast<size_t>(A_num_elements) * sizeof(float);

    const size_t B_bytes =
        static_cast<size_t>(B_num_elements) * sizeof(float);

    const size_t C_bytes =
        static_cast<size_t>(C_num_elements) * sizeof(float);


    // 申请主机内存
    float* A_host =
        static_cast<float*>(malloc(A_bytes));

    float* B_host =
        static_cast<float*>(malloc(B_bytes));

    float* C_cpu =
        static_cast<float*>(malloc(C_bytes));

    float* C_gpu_host =
        static_cast<float*>(malloc(C_bytes));


    // 检查主机内存是否申请成功
    if (
        A_host == nullptr ||
        B_host == nullptr ||
        C_cpu == nullptr ||
        C_gpu_host == nullptr
    )
    {
        printf("Failed to allocate host memory.\n");

        free(A_host);
        free(B_host);
        free(C_cpu);
        free(C_gpu_host);

        return 0;
    }


    // 初始化随机数种子
    srand(static_cast<unsigned int>(time(nullptr)));

    // 初始化输入矩阵
    random_init(A_host, A_num_elements);
    random_init(B_host, B_num_elements);


    // CPU 计算参考结果
    printf("Running CPU SGEMM...\n");

    sgemm_cpu(
        A_host,
        B_host,
        C_cpu,
        M,
        N,
        K
    );


    // 定义设备指针
    float* A_device = nullptr;
    float* B_device = nullptr;
    float* C_device = nullptr;


    // 申请 GPU 内存
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&A_device),
        A_bytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&B_device),
        B_bytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&C_device),
        C_bytes
    ));


    // 将输入矩阵复制到 GPU
    CHECK_CUDA(cudaMemcpy(
        A_device,
        A_host,
        A_bytes,
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        B_device,
        B_host,
        B_bytes,
        cudaMemcpyHostToDevice
    ));


    // 每个线程块包含 16 × 16 个线程
    dim3 threads(16, 16);

    // x 方向负责 C 的列
    // y 方向负责 C 的行
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );


    printf("Running GPU SGEMM v0...\n");

    // 启动 SGEMM kernel
    sgemm_v0_kernel<<<blocks, threads>>>(
        A_device,
        B_device,
        C_device,
        M,
        N,
        K
    );


    // 检查 kernel 启动是否出错
    CHECK_CUDA(cudaGetLastError());

    // 等待 kernel 执行完成
    CHECK_CUDA(cudaDeviceSynchronize());


    // 将 GPU 结果复制回主机
    CHECK_CUDA(cudaMemcpy(
        C_gpu_host,
        C_device,
        C_bytes,
        cudaMemcpyDeviceToHost
    ));


    // 比较 CPU 和 GPU 的计算结果
    bool correct = check_sgemm_result(
        C_cpu,
        C_gpu_host,
        M,
        N
    );

    if (correct)
    {
        printf("sgemm_v0 kernel result is correct.\n");
    }
    else
    {
        printf("sgemm_v0 kernel result is incorrect.\n");
    }


    // 释放 GPU 内存
    CHECK_CUDA(cudaFree(A_device));
    CHECK_CUDA(cudaFree(B_device));
    CHECK_CUDA(cudaFree(C_device));


    // 释放主机内存
    free(A_host);
    free(B_host);
    free(C_cpu);
    free(C_gpu_host);


    // 清理当前 CUDA 设备资源
    CHECK_CUDA(cudaDeviceReset());

    //return correct ? EXIT_SUCCESS : EXIT_FAILURE; 这啥意思，你return 0
    return 0;
}