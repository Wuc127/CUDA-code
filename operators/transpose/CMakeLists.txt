add_executable(transpose
    main.cu
    transpose_v0.cu
    transpose_v1.cu
    transpose_v2.cu
    transpose_v3.cu
    transpose_v4.cu
    transpose_cpu.cpp
    transpose_check.cpp
)

target_include_directories(transpose PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 各版本 Kernel 独立实现，没有跨文件的设备函数调用，无需可分离编译
set_target_properties(transpose PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RUNTIME_LIBRARY Shared
)
