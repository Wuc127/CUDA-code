# Softmax 算子

Softmax 算子将一组数值转换为概率分布。对于二维输入矩阵，通常对每一行进行计算：

$$
Y_{i,j} =
\frac{\exp(X_{i,j}-m_i)}
{\sum_k \exp(X_{i,k}-m_i)}
$$

其中，(m_i) 为第 (i) 行的最大值。计算时先减去最大值，可以提高数值稳定性，避免指数运算溢出。

## 文件说明

```text
softmax/
├── CMakeLists.txt
├── main.cu
├── softmax_cpu.cpp
├── softmax_cpu.h
├── softmax_check.cpp
├── softmax_check.h
├── softmax_v0.cu
├── softmax_v0.cuh
├── softmax_v1.cu
├── softmax_v1.cuh
├── softmax_v2.cu
├── softmax_v2.cuh
├── softmax_v3.cu
├── softmax_v3.cuh
├── softmax_v4.cu
└── softmax_v4.cuh
```

* `main.cu`：完成数据初始化、GPU 内存管理、算子调用、正确性检查和性能测试。
* `softmax_cpu.cpp`：CPU 参考实现。
* `softmax_check.cpp`：比较 CPU 和 GPU 的计算结果。
* `softmax_v*.cu`：不同版本的 CUDA 实现。
* `softmax_v*.cuh`：对应 CUDA kernel 的函数声明。

## softmax_v0

最朴素的实现方式。

* 一个线程负责计算一整行 Softmax。
* 依次计算最大值、指数和以及最终结果。
* 实现简单，但没有利用行内并行性。

## softmax_v1

使用一个线程块处理一行数据。

* 多个线程共同处理一行中的元素。
* 使用 Shared Memory 归约计算最大值和指数和。
* 提高了行内计算的并行度。

## softmax_v2

使用 Warp Shuffle 优化归约。

* 使用 `__shfl_down_sync` 完成 Warp 内归约。
* 不同 Warp 之间使用少量 Shared Memory 交换结果。
* 减少 Shared Memory 访问和线程同步开销。

## softmax_v3

使用 `float4` 进行向量化访存。

* 一个线程一次读取和处理四个连续的 `float` 元素。
* 结合 Warp Shuffle 计算最大值和指数和。
* 减少访存指令和索引计算次数。
* 单独处理元素数量不能被 4 整除的情况。

## softmax_v4

使用 Online Softmax。

* 在一次遍历中同时更新最大值和指数和。
* 减少对输入数据的重复读取。
* 结合 Warp Shuffle 完成线程间状态归约。
* 最后再次读取输入并计算归一化结果。

## 版本对比

| 版本 | 实现方式                    | 主要特点         |
| -- | ----------------------- | ------------ |
| v0 | 一个线程处理一行                | 实现简单，作为基础版本  |
| v1 | Shared Memory 归约        | 提高行内并行度      |
| v2 | Warp Shuffle 归约         | 减少同步和共享内存开销  |
| v3 | Warp Shuffle + `float4` | 使用向量化访存      |
| v4 | Online Softmax          | 合并最大值和指数和的计算 |

Softmax 包含最大值归约、求和归约、指数计算和全局内存访问，因此优化重点主要是提高行内并行度、减少线程同步、优化归约过程，并减少对输入数据的重复访问。
