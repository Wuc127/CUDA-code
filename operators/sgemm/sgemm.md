# SGEMM 算子优化

SGEMM 是单精度浮点矩阵乘法：

[
C = A \times B
]

其中：

* (A) 的形状为 (M \times K)
* (B) 的形状为 (K \times N)
* (C) 的形状为 (M \times N)

本项目通过多个 CUDA kernel，逐步学习 SGEMM 的优化方法。

## CPU Reference

使用 CPU 计算矩阵乘法，生成参考结果，用于检查 GPU kernel 的正确性。

对应文件：

```text
sgemm_cpu.cpp
sgemm_cpu.h
```

## SGEMM v0：Naive

一个线程计算矩阵 (C) 中的一个元素。

```text
一个线程 → 一个输出元素
```

特点：

* 直接从全局内存读取矩阵 (A) 和 (B)
* 实现简单
* 数据重复读取较多
* 作为最基础的性能基准

对应文件：

```text
sgemm_v0.cu
sgemm_v0.cuh
```

## SGEMM v1：Shared Memory Tiling

将矩阵划分为多个 tile，并把当前 tile 加载到共享内存中。

特点：

* 一个线程块计算一个输出 tile
* block 内线程共享输入数据
* 减少全局内存的重复访问
* 学习 `__shared__` 和 `__syncthreads()`

## SGEMM v2：一维寄存器分块

一个线程计算同一行或同一列中的多个输出元素。

```text
一个线程 → 多个输出元素
```

特点：

* 使用寄存器数组保存多个累加结果
* 提高共享内存数据的复用率
* 增加每个线程的计算量

## SGEMM v3：二维寄存器分块

一个线程计算一个较小的二维输出区域：

[
TM \times TN
]

特点：

* 使用二维寄存器数组保存结果
* 使用外积方式进行计算
* 同一组输入数据可以更新多个输出
* 是高性能 SGEMM 的核心优化方式

## SGEMM v4：向量化访存

使用 `float4` 等向量类型，一次读取或写入多个 `float`。

特点：

* 减少访存指令数量
* 提高全局内存访问效率
* 需要注意地址对齐
* 需要处理不能被 4 整除的矩阵尺寸

## SGEMM v5：Shared Memory Bank Conflict 优化

调整共享内存的数据布局，减少 bank conflict。

常见方法：

```text
转置存储
增加 padding
```

特点：

* 改善共享内存访问效率
* 学习 shared memory bank 的工作方式
* 性能提升取决于原本是否存在 bank conflict

## SGEMM v6：Double Buffering

使用两组共享内存缓冲区。

```text
计算当前 tile
同时准备下一个 tile
```

特点：

* 通过双缓冲实现软件流水线
* 尝试隐藏全局内存访问延迟
* 需要正确处理同步和缓冲区切换

GTX 1650 属于 Turing 架构，因此本版本先使用普通加载实现，不使用 Ampere 架构的 `cp.async`。

## SGEMM v7：Warp Tiling

在 block tile 和 thread tile 之间增加 warp tile。

```text
Block Tile
    └── Warp Tile
            └── Thread Tile
```

特点：

* 每个 warp 负责一个固定输出区域
* 明确 block、warp 和 thread 的任务划分
* 更接近成熟高性能 SGEMM 的实现结构
* 索引计算和参数设计更加复杂

## cuBLAS Baseline

使用 `cublasSgemm` 作为高性能参考。

作用：

* 对比自定义 kernel 与成熟库的性能差距
* 观察各个优化版本的性能提升
* cuBLAS 不负责生成 CPU 正确性参考结果

## 正确性检查

所有 GPU 版本都需要与 CPU 结果比较。

对应文件：

```text
sgemm_check.cpp
sgemm_check.h
```

由于浮点计算可能存在误差，通常使用绝对误差和相对误差进行判断。

## 性能测试

使用 CUDA Event 测量 kernel 的运行时间。

SGEMM 的浮点运算量近似为：

[
2MNK
]

GFLOPS 计算公式为：

[
GFLOPS =
\frac{2MNK}{t_{ms} \times 10^6}
]

其中 (t_{ms}) 是 kernel 的平均运行时间，单位为毫秒。

## 实现顺序

```text
CPU Reference
    ↓
v0 Naive
    ↓
v1 Shared Memory Tiling
    ↓
v2 一维寄存器分块
    ↓
v3 二维寄存器分块
    ↓
v4 向量化访存
    ↓
v5 Bank Conflict 优化
    ↓
v6 Double Buffering
    ↓
v7 Warp Tiling
    ↓
cuBLAS 性能对比
```

整个优化过程主要围绕以下几个方面展开：

```text
减少全局内存访问
提高数据复用率
增加寄存器计算量
优化共享内存访问
提高计算访存比
隐藏内存访问延迟
```
