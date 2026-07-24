# HGEMM 算子优化

HGEMM 用于计算半精度矩阵乘法：

[
C = A \times B
]

其中矩阵 `A`、`B` 和 `C` 使用 FP16 数据类型。为了提高计算精度，各版本通常使用 FP32 进行中间累加，并在计算完成后将结果转换为 FP16。

## 文件说明

```text
hgemm/
├── main.cu
├── hgemm_cpu.cpp
├── hgemm_cpu.h
├── hgemm_check.cpp
├── hgemm_check.h
├── hgemm_v0.cu
├── hgemm_v0.cuh
├── hgemm_v1.cu
├── hgemm_v1.cuh
├── hgemm_v2.cu
├── hgemm_v2.cuh
├── hgemm_v3.cu
├── hgemm_v3.cuh
├── hgemm_v4.cu
├── hgemm_v4.cuh
├── hgemm_v5.cu
└── hgemm_v5.cuh
```

* `main.cu`：完成数据初始化、GPU 内存管理、算子调用、正确性检查和性能测试。
* `hgemm_cpu.cpp`：CPU 参考实现，使用 FP32 进行乘法和累加。
* `hgemm_check.cpp`：比较 CPU 和 GPU 的计算结果。
* `hgemm_v*.cu`：不同版本的 HGEMM CUDA 实现。
* `hgemm_v*.cuh`：对应 CUDA kernel 的函数声明。

## HGEMM v0：朴素实现

一个 CUDA 线程计算输出矩阵 `C` 中的一个元素。

每个线程遍历矩阵的 `K` 维度，依次读取 `A` 的一行和 `B` 的一列，并完成点积计算。

该版本没有使用共享内存，存在较多重复的全局内存访问，主要作为后续优化版本的性能基准。

## HGEMM v1：共享内存分块

将矩阵划分为多个小块，并将当前计算所需的 `A` 和 `B` 子矩阵加载到共享内存中。

线程块内的多个线程可以重复使用共享内存中的数据，从而减少全局内存访问次数，提高数据访问效率。

## HGEMM v2：寄存器分块

在共享内存分块的基础上，让每个线程计算多个输出元素。

每个线程使用寄存器保存多个累加结果，从而提高数据复用率，并减少线程数量和共享内存访问次数。

## HGEMM v3：Half2 向量化

使用 `__half2` 数据类型一次加载和处理两个 FP16 数据。

向量化访问可以减少内存访问指令数量，并提高 FP16 数据的计算吞吐率。

为了保证计算精度，可以将读取的 FP16 数据转换为 FP32 后进行累加。

## HGEMM v4：综合优化

综合使用以下优化方法：

* 共享内存分块；
* 寄存器分块；
* Half2 向量化访问；
* 数据预取；
* 软件双缓冲。

在计算当前数据块的同时加载下一个数据块，以减少数据加载与计算之间的等待时间。

## HGEMM v5：WMMA 实现

使用 CUDA WMMA API 调用 Tensor Core 完成矩阵乘法。

输入矩阵通常使用 FP16，累加矩阵使用 FP32。WMMA 以固定大小的矩阵块为基本计算单位，例如 `16 × 16 × 16`。

该版本主要用于学习 Tensor Core 的编程方式，需要支持 Tensor Core 的 NVIDIA GPU 才能获得明显的性能提升。

## 优化顺序

```text
v0：朴素矩阵乘法
 ↓
v1：共享内存分块
 ↓
v2：寄存器分块
 ↓
v3：Half2 向量化
 ↓
v4：综合优化与双缓冲
 ↓
v5：WMMA / Tensor Core
```

通过逐步实现这些版本，可以观察不同优化方法对 HGEMM 性能的影响，并理解 CUDA 矩阵乘法算子的基本优化思路。
