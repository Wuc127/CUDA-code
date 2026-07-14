# CUDA-code

这是一个用于学习 CUDA 算子实现与优化的项目。项目目标是从基础算子开始，逐步实现和优化常见深度学习算子，例如 reduce、SGEMM、HGEMM、softmax、layernorm、attention 和 flash attention 等。

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
└─ flash_attention/
```

这里不再额外划分 `basic/`、`linear_algebra/`、`transformer/` 等大类。
当前阶段采用“一个算子一个目录”的方式，更清楚，也更方便学习和迭代。


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

推荐命名方式：

```text
reduce_v0
sgemm_v0
hgemm_v0
softmax_v0
layernorm_v0
attention_v0
flash_attention_v0
```

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
