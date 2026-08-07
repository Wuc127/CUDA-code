# Transpose 算子

Transpose 算子用于交换矩阵的行和列。对于输入矩阵 (X)，其转置结果 (Y) 定义为：

$$
Y_{j,i}=X_{i,j}
$$

其中，输入矩阵大小为 (M \times N)，输出矩阵大小为 (N \times M)。

## 文件说明

```text
transpose/
├── CMakeLists.txt
├── main.cu
├── transpose_cpu.cpp
├── transpose_cpu.h
├── transpose_check.cpp
├── transpose_check.h
├── transpose_v0.cu
├── transpose_v0.cuh
├── transpose_v1.cu
├── transpose_v1.cuh
├── transpose_v2.cu
├── transpose_v2.cuh
├── transpose_v3.cu
├── transpose_v3.cuh
├── transpose_v4.cu
└── transpose_v4.cuh
```

* `main.cu`：完成数据初始化、GPU 内存管理、算子调用、正确性检查和性能测试。
* `transpose_cpu.cpp`：CPU 参考实现。
* `transpose_check.cpp`：比较 CPU 和 GPU 的计算结果。
* `transpose_v*.cu`：不同版本的 CUDA 实现。
* `transpose_v*.cuh`：对应 CUDA kernel 的函数声明。

## transpose_v0

最朴素的 Global Memory 实现。

* 一个线程负责转置一个矩阵元素。
* 直接从 Global Memory 读取输入并写入输出。
* 输入读取连续，但输出写入通常是不连续的。
* 实现简单，作为基础版本。

## transpose_v1

使用 Shared Memory 进行分块转置。

* 一个线程块负责处理一个矩阵 Tile。
* 先将数据从 Global Memory 读取到 Shared Memory。
* 在线程块内部完成转置后再写入 Global Memory。
* 使输入读取和输出写入都能够实现更好的合并访存。

## transpose_v2

通过 Padding 消除 Shared Memory Bank Conflict。

* Shared Memory Tile 定义为 `TILE_DIM × (TILE_DIM + 1)`。
* 额外增加一列 Padding。
* 避免转置访问 Shared Memory 时出现严重的 Bank Conflict。
* 在 v1 的基础上进一步提高访存效率。

## transpose_v3

使用 `float4` 进行向量化访存。

* 一个线程一次读取或写入四个连续的 `float` 元素。
* 减少 Global Memory 访存指令数量。
* 结合 Shared Memory 和 Padding 完成矩阵转置。
* 对不能被 4 整除的边界元素进行单独处理。

## transpose_v4

优化 Tile 和线程映射方式。

* 一个线程负责处理 Tile 中的多个元素。
* 使用较少线程完成较大的 Tile 数据搬运。
* 减少线程块调度和索引计算开销。
* 结合 Shared Memory、Padding 和向量化访存进一步优化性能。

## 版本对比

| 版本 | 实现方式 | 主要特点 |
| -- | ----------------------- | ------------------- |
| v0 | Global Memory | 最基础的直接转置 |
| v1 | Shared Memory Tile | 改善 Global Memory 访存 |
| v2 | Shared Memory + Padding | 消除 Bank Conflict |
| v3 | `float4` 向量化访存 | 减少访存指令 |
| v4 | 优化 Tile 与线程映射 | 提高数据搬运效率 |

Transpose 算子的计算量很小，性能主要受内存访问效率影响，因此优化重点主要是改善 Global Memory 合并访存、减少 Shared Memory Bank Conflict，并提高数据搬运效率。
