cmake_minimum_required(VERSION 3.18)

project(cuda_code LANGUAGES CXX CUDA)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 你的 GTX 1650 是 sm_75
set(CMAKE_CUDA_ARCHITECTURES 75)

# 让 CUDA 代码支持可分离编译，后面多个 .cu 文件时更稳
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Windows / MSVC 下减少一些无关警告
if (MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

# 添加 operators 下面的 attention 子目录
add_subdirectory(operators/attention)