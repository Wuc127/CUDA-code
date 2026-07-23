# CUDA-code

### CUDA 高性能计算与算子优化

<p>
  <img src="https://img.shields.io/badge/CUDA-12.3-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=cplusplus&logoColor=white" alt="C++">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/CMake-Build-064F8C?style=for-the-badge&logo=cmake&logoColor=white" alt="CMake">
  <img src="https://img.shields.io/badge/Visual%20Studio-2022-5C2D91?style=for-the-badge&logo=visualstudio&logoColor=white" alt="Visual Studio">
</p>

---

## 技术栈

<p>
  <img src="https://skillicons.dev/icons?i=cpp,cmake,visualstudio,git,github" alt="Technology Stack">
</p>

* **CUDA C++**：GPU 并行计算与 CUDA Kernel 开发
* **C++**：主机端代码、测试程序与算子封装
* **CMake**：项目构建与目录管理
* **Python**：算子验证、性能测试，以及 Triton 实现与对比
* **Visual Studio / VS Code**：代码编写、编译与调试
* **Git / GitHub**：代码版本管理

CUDA 算子实现与优化，从基础算子开始，逐步实现和优化常见深度学习算子，例如 reduce、SGEMM、HGEMM、softmax、layernorm、attention、flash attention 等。

## 1.目录结构

```text
CUDA-code/
│
├─ CMakeLists.txt
│
├─ README.md
│
├─ .vscode/
│  └─ settings.json
│
├─ build/
│
├─ operators/
│  ├─ attention/
│  ├─ sgemm/
│  ├─ hgemm/
│  ├─ reduce/
│  ├─ softmax/
│  ├─ layernorm/
│  └─ flash_attention/
│
└─ .gitignore/
```

## 2.目录说明

### 顶层目录

```text
CUDA-code/
```

项目根目录，存放顶层 `CMakeLists.txt`、`README.md`、VS Code 配置以及所有算子目录。

### CMakeLists.txt

顶层 CMake 配置文件，负责设置整个项目的 C++ / CUDA 编译选项，并通过 `add_subdirectory()` 添加各个算子子目录。
每添加一个新的算子目录，都需要在顶层 `CMakeLists.txt` 中注册对应子目录。

### .vscode/settings.json

VS Code 当前项目的配置文件，用于指定 CMake Tools 的生成器、平台、工具集和 build 目录。

当前项目使用：

```text
Visual Studio 17 2022
x64
v142
MSVC 14.29.30133
```

### operators/

存放所有算子实现。每个算子单独一个目录。

当前推荐组织方式是：

```text
operators/
├─ attention/
├─ sgemm/
├─ hgemm/
├─ reduce/
├─ softmax/
├─ layernorm/
└─ elementwise/
```

## 3.CMake target 命名规则

每个算子目录中可以通过 `add_executable()` 创建可执行目标。

例如：

```cmake
add_executable(attention_v0
    main.cu
    attention_v0.cu
    attention_cpu.cpp
    check_result.cpp
)
```

其中：

```text
attention_v0
```

就是 CMake target 名字。

构建指定目标时使用：

```bash
cmake --build build --config Debug --target attention_v0
```

注意：target 名字在整个 CMake 项目中必须全局唯一。即使不同算子位于不同目录下，也不能创建同名 target。


## 4.构建方式

项目使用 CMake 构建。

在 VS Code 中推荐使用 CMake Tools：

```text
CMake: Configure
CMake: Build
CMake: Build Target
CMake: Set Launch/Debug Target
```

如果使用命令行，构建整个项目：

```bash
cmake --build build --config Debug
```

只构建某个目标：

```bash
cmake --build build --config Debug --target attention_v0
```

## 5.运行方式

构建成功后，可执行文件会生成在类似路径：

```text
build/operators/attention/Debug/attention_v0.exe
```

可以直接在终端运行：

```bash
build/operators/attention/Debug/attention_v0.exe
```

也可以在 VS Code 中：

```text
CMake: Set Launch/Debug Target
→ attention_v0
```

然后使用底部按钮：

```text
在终端窗口中启动所选目标
```

普通运行程序。

如果需要断点调试，可以使用：

```text
启动所选目标的调试程序
```
