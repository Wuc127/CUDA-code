# LayerNorm 算子

LayerNorm 对每个样本的特征进行归一化。对于二维输入矩阵，通常对每一行进行计算：

$$
Y_{i,j}
=
\frac{X_{i,j}-\mu_i}
{\sqrt{\sigma_i^2+\epsilon}}
\cdot \gamma_j+\beta_j
$$

其中，(\mu_i) 和 (\sigma_i^2) 分别表示第 (i) 行的均值和方差，(\gamma) 和 (\beta) 为可学习参数，(\epsilon) 用于避免除零。

## 文件说明

```text
layernorm/
├── CMakeLists.txt
├── main.cu
├── layernorm_cpu.cpp
├── layernorm_cpu.h
├── layernorm_check.cpp
├── layernorm_check.h
├── layernorm_v0.cu
├── layernorm_v0.cuh
├── layernorm_v1.cu
├── layernorm_v1.cuh
├── layernorm_v2.cu
├── layernorm_v2.cuh
├── layernorm_v3.cu
├── layernorm_v3.cuh
├── layernorm_v4.cu
└── layernorm_v4.cuh
```

* `main.cu`：完成数据初始化、GPU 内存管理、算子调用、正确性检查和性能测试。
* `layernorm_cpu.cpp`：CPU 参考实现。
* `layernorm_check.cpp`：比较 CPU 和 GPU 的计算结果。
* `layernorm_v*.cu`：不同版本的 CUDA 实现。
* `layernorm_v*.cuh`：对应 CUDA kernel 的函数声明。

## layernorm_v0

最朴素的实现方式。

* 一个线程负责计算一整行 LayerNorm。
* 依次计算均值、方差和归一化结果。
* 实现简单，但没有利用行内并行性。

## layernorm_v1

使用一个线程块处理一行数据。

* 多个线程共同处理一行中的元素。
* 使用 Shared Memory 归约计算均值和方差。
* 提高行内计算的并行度。

## layernorm_v2

使用 Warp Shuffle 优化归约。

* 使用 `__shfl_down_sync` 完成 Warp 内求和归约。
* 不同 Warp 之间使用少量 Shared Memory 交换结果。
* 减少 Shared Memory 访问和线程同步开销。

## layernorm_v3

使用 Welford 算法计算均值和方差。

* 在一次遍历中同时更新均值和方差。
* 提高方差计算的数值稳定性。
* 在线程之间合并局部 Welford 状态。

## layernorm_v4

结合 Welford 算法和 `float4` 向量化访存。

* 一个线程一次读取和处理四个连续的 `float` 元素。
* 使用 Welford 算法计算均值和方差。
* 使用 Warp Shuffle 完成线程间归约。
* 减少访存指令和索引计算次数。

## 版本对比

| 版本 | 实现方式               | 主要特点         |
| -- | ------------------ | ------------ |
| v0 | 一个线程处理一行           | 实现简单，作为基础版本  |
| v1 | Shared Memory 归约   | 提高行内并行度      |
| v2 | Warp Shuffle 归约    | 减少同步和共享内存开销  |
| v3 | Welford 算法         | 提高数值稳定性      |
| v4 | Welford + `float4` | 使用向量化访存进一步优化 |

LayerNorm 包含均值归约、方差归约、开方和全局内存访问，因此优化重点主要是提高行内并行度、优化归约过程、减少线程同步，并提高显存访问效率。
