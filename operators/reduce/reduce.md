# reduce 算子优化



## 文件说明

```text
reduce/
├── main.cu
├── reduce_cpu.cpp
├── reduce_cpu.h
├── reduce_check.cpp
├── reduce_check.h
├── reduce_v0.cu
├── reduce_v0.cuh
├── reduce_v1.cu
├── reduce_v1.cuh
├── reduce_v2.cu
├── reduce_v2.cuh
├── reduce_v3.cu
├── reduce_v3.cuh
├── reduce_v4.cu
├── reduce_v4.cuh
├── reduce_v5.cu
├── reduce_v5.cuh
├── reduce_v6.cu
├── reduce_v6.cuh
├── reduce_v7.cu
└── reduce_v7.cuh
```

* `main.cu`：完成数据初始化、GPU 内存管理、算子调用、正确性检查和性能测试。
* `reduce_cpu.cpp`：CPU 参考实现
* `reduce_check.cpp`：比较 CPU 和 GPU 的计算结果。
* `reduce_v*.cu`：不同版本的 CUDA 实现。
* `reduce_v*.cuh`：对应 CUDA kernel 的函数声明。
