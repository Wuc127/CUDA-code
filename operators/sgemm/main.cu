#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "sgemm_check.h"
#include "sgemm_cpu.h"
#include "sgemm_cublas.cuh"
#include "sgemm_v0.cuh"
#include "sgemm_v1.cuh"
#include "sgemm_v2.cuh"
#include "sgemm_v3.cuh"
#include "sgemm_v4.cuh"
#include "sgemm_v5.cuh"
#include "sgemm_v6.cuh"
#include "sgemm_v7.cuh"


const char* cublas_status_to_string(cublasStatus_t status)
{
    switch (status)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "Unknown cuBLAS status";
    }
}


#define CHECK_CUDA(call)                                               \
    do                                                                 \
    {                                                                  \
        const cudaError_t cuda_error = (call);                          \
        if (cuda_error != cudaSuccess)                                 \
        {                                                              \
            std::printf("CUDA error at %s:%d\n", __FILE__, __LINE__); \
            std::printf("Error: %s\n", cudaGetErrorString(cuda_error)); \
            std::exit(EXIT_FAILURE);                                   \
        }                                                              \
    } while (0)


#define CHECK_CUBLAS(call)                                               \
    do                                                                   \
    {                                                                    \
        const cublasStatus_t cublas_status = (call);                      \
        if (cublas_status != CUBLAS_STATUS_SUCCESS)                       \
        {                                                                \
            std::printf("cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
            std::printf("Error: %s\n", cublas_status_to_string(cublas_status)); \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                \
    } while (0)


void random_init(float* data, std::size_t num_elements)
{
    for (std::size_t i = 0; i < num_elements; ++i)
    {
        data[i] =
            2.0f * static_cast<float>(std::rand()) /
            static_cast<float>(RAND_MAX) - 1.0f;
    }
}


int ceil_div(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}


template <typename Launcher>
bool run_sgemm_test(
    const char* name,
    Launcher launch,
    const float* C_cpu,
    float* C_device,
    float* C_gpu_host,
    int M,
    int N,
    int K,
    std::size_t C_bytes,
    int warmup_iterations,
    int benchmark_iterations
)
{
    std::printf("\nRunning %s...\n", name);

    CHECK_CUDA(cudaMemset(C_device, 0, C_bytes));
    launch();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(
        C_gpu_host,
        C_device,
        C_bytes,
        cudaMemcpyDeviceToHost
    ));

    const bool correct = check_sgemm_result(C_cpu, C_gpu_host, M, N);

    if (!correct)
    {
        std::printf("%s result is incorrect; benchmark skipped.\n", name);
        return false;
    }

    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
    {
        launch();
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int iteration = 0; iteration < benchmark_iterations; ++iteration)
    {
        launch();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float total_milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_milliseconds, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    const float average_milliseconds =
        total_milliseconds / static_cast<float>(benchmark_iterations);
    const double operations =
        2.0 * static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);
    const double gflops =
        operations / (static_cast<double>(average_milliseconds) * 1.0e6);

    std::printf("%s result is correct.\n", name);
    std::printf("Average time: %.6f ms\n", average_milliseconds);
    std::printf("Performance:  %.2f GFLOPS\n", gflops);

    return true;
}


int main()
{
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;
    constexpr int warmup_iterations = 5;
    constexpr int benchmark_iterations = 50;

    const std::size_t A_num_elements =
        static_cast<std::size_t>(M) * static_cast<std::size_t>(K);
    const std::size_t B_num_elements =
        static_cast<std::size_t>(K) * static_cast<std::size_t>(N);
    const std::size_t C_num_elements =
        static_cast<std::size_t>(M) * static_cast<std::size_t>(N);

    const std::size_t A_bytes = A_num_elements * sizeof(float);
    const std::size_t B_bytes = B_num_elements * sizeof(float);
    const std::size_t C_bytes = C_num_elements * sizeof(float);

    std::vector<float> A_host(A_num_elements);
    std::vector<float> B_host(B_num_elements);
    std::vector<float> C_cpu(C_num_elements);
    std::vector<float> C_gpu_host(C_num_elements);

    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    random_init(A_host.data(), A_num_elements);
    random_init(B_host.data(), B_num_elements);

    std::printf("Matrix size: M = %d, N = %d, K = %d\n", M, N, K);
    std::printf(
        "Warmup iterations: %d, benchmark iterations: %d\n",
        warmup_iterations,
        benchmark_iterations
    );
    std::printf("\nRunning CPU reference SGEMM...\n");

    sgemm_cpu(A_host.data(), B_host.data(), C_cpu.data(), M, N, K);

    float* A_device = nullptr;
    float* B_device = nullptr;
    float* C_device = nullptr;

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&A_device), A_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&B_device), B_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&C_device), C_bytes));

    CHECK_CUDA(cudaMemcpy(
        A_device,
        A_host.data(),
        A_bytes,
        cudaMemcpyHostToDevice
    ));
    CHECK_CUDA(cudaMemcpy(
        B_device,
        B_host.data(),
        B_bytes,
        cudaMemcpyHostToDevice
    ));

    cublasHandle_t cublas_handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    const auto launch_v0 = [&]()
    {
        const dim3 threads(16, 16);
        const dim3 blocks(ceil_div(N, 16), ceil_div(M, 16));
        sgemm_v0_kernel<<<blocks, threads>>>(
            A_device, B_device, C_device, M, N, K
        );
    };

    const auto launch_v1 = [&]()
    {
        const dim3 threads(SGEMM_V1_TILE_SIZE, SGEMM_V1_TILE_SIZE);
        const dim3 blocks(
            ceil_div(N, SGEMM_V1_TILE_SIZE),
            ceil_div(M, SGEMM_V1_TILE_SIZE)
        );
        sgemm_v1_kernel<<<blocks, threads>>>(
            A_device, B_device, C_device, M, N, K
        );
    };

    const auto launch_v2 = [&]()
    {
        const dim3 threads(SGEMM_V2_NUM_THREADS);
        const dim3 blocks(
            ceil_div(N, SGEMM_V2_BLOCK_TILE_N),
            ceil_div(M, SGEMM_V2_BLOCK_TILE_M)
        );
        sgemm_v2_kernel<<<blocks, threads>>>(
            A_device, B_device, C_device, M, N, K
        );
    };

    const auto launch_v3 = [&]()
    {
        const dim3 threads(SGEMM_V3_NUM_THREADS);
        const dim3 blocks(
            ceil_div(N, SGEMM_V3_BLOCK_TILE_N),
            ceil_div(M, SGEMM_V3_BLOCK_TILE_M)
        );
        sgemm_v3_kernel<<<blocks, threads>>>(
            A_device, B_device, C_device, M, N, K
        );
    };

    const auto launch_v4 = [&]()
    {
        const dim3 threads(SGEMM_V4_NUM_THREADS);
        const dim3 blocks(
            ceil_div(N, SGEMM_V4_BLOCK_TILE_N),
            ceil_div(M, SGEMM_V4_BLOCK_TILE_M)
        );
        sgemm_v4_kernel<<<blocks, threads>>>(
            A_device, B_device, C_device, M, N, K
        );
    };

    const auto launch_v5 = [&]()
    {
        const dim3 threads(SGEMM_V5_NUM_THREADS);
        const dim3 blocks(
            ceil_div(N, SGEMM_V5_BLOCK_TILE_N),
            ceil_div(M, SGEMM_V5_BLOCK_TILE_M)
        );
        sgemm_v5_kernel<<<blocks, threads>>>(
            A_device, B_device, C_device, M, N, K
        );
    };

    const auto launch_v6 = [&]()
    {
        const dim3 threads(SGEMM_V6_NUM_THREADS);
        const dim3 blocks(
            ceil_div(N, SGEMM_V6_BLOCK_TILE_N),
            ceil_div(M, SGEMM_V6_BLOCK_TILE_M)
        );
        sgemm_v6_kernel<<<blocks, threads>>>(
            A_device, B_device, C_device, M, N, K
        );
    };

    const auto launch_v7 = [&]()
    {
        const dim3 threads(SGEMM_V7_NUM_THREADS);
        const dim3 blocks(
            ceil_div(N, SGEMM_V7_BLOCK_TILE_N),
            ceil_div(M, SGEMM_V7_BLOCK_TILE_M)
        );
        sgemm_v7_kernel<<<blocks, threads>>>(
            A_device, B_device, C_device, M, N, K
        );
    };

    const auto launch_cublas = [&]()
    {
        CHECK_CUBLAS(sgemm_cublas(
            cublas_handle,
            A_device,
            B_device,
            C_device,
            M,
            N,
            K
        ));
    };

    bool all_correct = true;

    const auto run_test = [&](const char* name, auto launch)
    {
        const bool correct = run_sgemm_test(
            name,
            launch,
            C_cpu.data(),
            C_device,
            C_gpu_host.data(),
            M,
            N,
            K,
            C_bytes,
            warmup_iterations,
            benchmark_iterations
        );

        all_correct = correct && all_correct;
    };

    run_test("sgemm_v0", launch_v0);
    run_test("sgemm_v1", launch_v1);
    run_test("sgemm_v2", launch_v2);
    run_test("sgemm_v3", launch_v3);
    run_test("sgemm_v4", launch_v4);
    run_test("sgemm_v5", launch_v5);
    run_test("sgemm_v6", launch_v6);
    run_test("sgemm_v7", launch_v7);
    run_test("cuBLAS", launch_cublas);

    std::printf(
        "\nOverall result: %s\n",
        all_correct
            ? "all implementations passed"
            : "one or more implementations failed"
    );

    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    // 释放 GPU 内存
    CHECK_CUDA(cudaFree(A_device));
    CHECK_CUDA(cudaFree(B_device));
    CHECK_CUDA(cudaFree(C_device));

    // 清理当前 CUDA 设备资源
    CHECK_CUDA(cudaDeviceReset());

    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
