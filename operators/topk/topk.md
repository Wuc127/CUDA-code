# Top-K 算子

Top-K 算子用于从一组数据中选出最大的 (K) 个元素及其对应索引。对于二维输入矩阵，通常对每一行分别进行计算：

$$
(V_i, I_i)=\operatorname{TopK}(X_i,K)
$$

其中，(V_i) 表示第 (i) 行中最大的 (K) 个元素，(I_i) 表示这些元素在原始行中的索引。

## 文件说明

```text
topk/
├── CMakeLists.txt
├── main.cu
├── topk_cpu.cpp
├── topk_cpu.h
├── topk_check.cpp
├── topk_check.h
├── topk_v0.cu
├── topk_v0.cuh
├── topk_v1.cu
├── topk_v1.cuh
├── topk_v2.cu
├── topk_v2.cuh
├── topk_v3.cu
├── topk_v3.cuh
├── topk_v4.cu
└── topk_v4.cuh
```

* `main.cu`：完成数据初始化、GPU 内存管理、算子调用、正确性检查和性能测试。
* `topk_cpu.cpp`：CPU 参考实现。
* `topk_check.cpp`：比较 CPU 和 GPU 输出的 Top-K 数值及索引。
* `topk_v*.cu`：不同版本的 CUDA 实现。
* `topk_v*.cuh`：对应 CUDA kernel 的函数声明。

## topk_v0

最朴素的 Global Memory 实现。

* 一个线程负责处理一整行数据。
* 每次从 Global Memory 中遍历整行并寻找最大值。
* 找到一个最大值后将其标记，再继续寻找下一个最大值。
* 总共重复 (K) 次，实现简单但并行度较低。

## topk_v1

使用 Shared Memory 缓存输入数据。

* 一个线程块负责处理一行数据。
* 多个线程将输入从 Global Memory 加载到 Shared Memory。
* 后续的 Top-K 查找直接访问 Shared Memory。
* 减少重复访问 Global Memory 的开销。

## topk_v2

使用 Shared Memory 并行归约。

* 多个线程共同寻找当前最大值。
* 使用 Shared Memory 归约得到最大值及其索引。
* 每找到一个最大值后，将对应元素设置为负无穷。
* 重复执行 (K) 次归约得到最终结果。

## topk_v3

使用 Warp Shuffle 优化归约。

* 使用 `__shfl_down_sync` 完成 Warp 内最大值归约。
* 同时传递元素数值和对应索引。
* 不同 Warp 之间使用少量 Shared Memory 合并结果。
* 减少 Shared Memory 访问和线程同步开销。

## topk_v4

使用线程局部 Top-K 和候选集合合并。

* 每个线程处理多个输入元素。
* 每个线程在寄存器中维护自己的局部 Top-K。
* 将各线程的局部候选结果写入 Shared Memory。
* 对候选集合进行合并，得到整行最终的 Top-K。
* 避免对完整输入重复执行 (K) 次归约。

## 版本对比

| 版本 | 实现方式 | 主要特点 |
| -- | ------------------ | ----------- |
| v0 | Global Memory 串行查找 | 实现简单，作为基础版本 |
| v1 | Shared Memory 缓存 | 减少重复全局内存访问 |
| v2 | Shared Memory 并行归约 | 提高最大值查找并行度 |
| v3 | Warp Shuffle 归约 | 减少同步和共享内存开销 |
| v4 | 局部 Top-K + 候选合并 | 减少重复遍历和归约次数 |

Top-K 包含最大值查找、索引维护、归约和候选结果合并，因此优化重点主要是提高行内并行度、减少 Global Memory 访问、优化归约过程，并减少对完整输入数据的重复遍历。
