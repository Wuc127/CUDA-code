#include "sgemm_cublas.cuh"


cublasStatus_t sgemm_cublas(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta
)
{
    // cuBLAS 默认使用列主序，而当前项目中的矩阵使用行主序。
    // 行主序的 C = A * B 等价于列主序的 C^T = B^T * A^T，
    // 因此交换 A 和 B 的传入顺序，并交换 M 和 N。
    return cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        B,
        N,
        A,
        K,
        &beta,
        C,
        N
    );
}
