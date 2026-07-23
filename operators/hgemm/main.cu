#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "hgemm_cpu.h"
#include "hgemm_check.h"
#include "hgemm_v0.cuh"


// 检查 CUDA API 是否调用成功
#define CHECK_CUDA(call)                                         \
    do                                                           \
    {                                                            \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess)                                  \
        {                                                        \
            std::printf(                                         \
                "CUDA error at %s:%d\n",                          \
                __FILE__,                                        \
                __LINE__                                         \
            );                                                   \
            std::printf(                                         \
                "Error: %s\n",                                   \
                cudaGetErrorString(err)                           \
            );                                                   \
            std::exit(EXIT_FAILURE);                             \
        }                                                        \
    } while (0)


// 随机初始化矩阵，生成 [-1, 1] 范围内的数据
void random_init(__half* data, int size)
{
    for (int i = 0; i < size; i++)
    {
        float value =
            static_cast<float>(std::rand()) /
            static_cast<float>(RAND_MAX);

        value = value * 2.0f - 1.0f;

        data[i] = __float2half(value);
    }
}


int main()
{
    // 矩阵大小：
    //
    // A: [M, K]
    // B: [K, N]
    // C: [M, N]
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    const int warmup_iterations = 5;
    const int test_iterations = 20;

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    int num_elements_A = M * K;
    int num_elements_B = K * N;
    int num_elements_C = M * N;

    size_t bytes_A =
        static_cast<size_t>(num_elements_A) * sizeof(__half);

    size_t bytes_B =
        static_cast<size_t>(num_elements_B) * sizeof(__half);

    size_t bytes_C_half =
        static_cast<size_t>(num_elements_C) * sizeof(__half);

    size_t bytes_C_float =
        static_cast<size_t>(num_elements_C) * sizeof(float);


    // ============================
    // 申请主机内存
    // ============================

    __half* A_host =
        static_cast<__half*>(std::malloc(bytes_A));

    __half* B_host =
        static_cast<__half*>(std::malloc(bytes_B));

    __half* C_gpu_host =
        static_cast<__half*>(std::malloc(bytes_C_half));

    float* C_cpu =
        static_cast<float*>(std::malloc(bytes_C_float));


    if (A_host == nullptr ||
        B_host == nullptr ||
        C_gpu_host == nullptr ||
        C_cpu == nullptr)
    {
        std::printf("Host memory allocation failed.\n");

        std::free(A_host);
        std::free(B_host);
        std::free(C_gpu_host);
        std::free(C_cpu);

        return 1;
    }


    // ============================
    // 初始化输入矩阵
    // ============================

    random_init(A_host, num_elements_A);
    random_init(B_host, num_elements_B);


    // ============================
    // CPU 参考计算
    // ============================

    std::printf("Computing CPU reference result...\n");

    hgemm_cpu(
        A_host,
        B_host,
        C_cpu,
        M,
        N,
        K
    );

    std::printf("CPU reference computation completed.\n");


    // ============================
    // 申请 GPU 内存
    // ============================

    __half* A_device = nullptr;
    __half* B_device = nullptr;
    __half* C_device = nullptr;

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&A_device),
        bytes_A
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&B_device),
        bytes_B
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void**>(&C_device),
        bytes_C_half
    ));


    // ============================
    // 将输入数据复制到 GPU
    // ============================

    CHECK_CUDA(cudaMemcpy(
        A_device,
        A_host,
        bytes_A,
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemcpy(
        B_device,
        B_host,
        bytes_B,
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA(cudaMemset(
        C_device,
        0,
        bytes_C_half
    ));


    // ============================
    // 设置 kernel 启动参数
    // ============================

    dim3 threads(16, 16);

    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );


    // ============================
    // 首次运行，检查 kernel
    // ============================

    hgemm_v0_kernel<<<blocks, threads>>>(
        A_device,
        B_device,
        C_device,
        M,
        N,
        K
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    // ============================
    // 将 GPU 结果复制回主机
    // ============================

    CHECK_CUDA(cudaMemcpy(
        C_gpu_host,
        C_device,
        bytes_C_half,
        cudaMemcpyDeviceToHost
    ));


    // ============================
    // 检查计算结果
    // ============================

    bool correct = check_hgemm_result(
        C_cpu,
        C_gpu_host,
        M,
        N,
        5e-2f,
        5e-2f
    );


    // ============================
    // 预热
    // ============================

    for (int i = 0; i < warmup_iterations; i++)
    {
        hgemm_v0_kernel<<<blocks, threads>>>(
            A_device,
            B_device,
            C_device,
            M,
            N,
            K
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    // ============================
    // 性能测试
    // ============================

    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < test_iterations; i++)
    {
        hgemm_v0_kernel<<<blocks, threads>>>(
            A_device,
            B_device,
            C_device,
            M,
            N,
            K
        );
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_time_ms = 0.0f;

    CHECK_CUDA(cudaEventElapsedTime(
        &total_time_ms,
        start,
        stop
    ));

    float average_time_ms =
        total_time_ms /
        static_cast<float>(test_iterations);


    // 矩阵乘法总浮点运算次数：
    //
    // M × N 个输出元素；
    // 每个输出元素执行 K 次乘法和 K 次加法。
    double total_operations =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);

    // TFLOPS = 运算次数 / 时间 / 10^12
    //
    // average_time_ms 需要转换为秒，因此：
    //
    // TFLOPS = operations / (ms × 10^-3) / 10^12
    //         = operations / ms / 10^9
    double tflops =
        total_operations /
        static_cast<double>(average_time_ms) /
        1.0e9;


    std::printf("\n");
    std::printf("HGEMM v0 performance:\n");
    std::printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    std::printf("Block size: %d x %d\n", threads.x, threads.y);
    std::printf("Grid size: %d x %d\n", blocks.x, blocks.y);
    std::printf("Average time: %.6f ms\n", average_time_ms);
    std::printf("Performance: %.6f TFLOPS\n", tflops);


    // ============================
    // 释放资源
    // ============================

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(A_device));
    CHECK_CUDA(cudaFree(B_device));
    CHECK_CUDA(cudaFree(C_device));

    std::free(A_host);
    std::free(B_host);
    std::free(C_gpu_host);
    std::free(C_cpu);

    CHECK_CUDA(cudaDeviceReset());

    return correct ? 0 : 1;
}