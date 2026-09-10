find_package(CUDAToolkit REQUIRED)

add_executable(sgemm_v0
    main.cu
    sgemm_check.cpp
    sgemm_cpu.cpp
    sgemm_cublas.cu
    sgemm_v0.cu
    sgemm_v1.cu
    sgemm_v2.cu
    sgemm_v3.cu
    sgemm_v4.cu
    sgemm_v5.cu
    sgemm_v6.cu
    sgemm_v7.cu
)

target_include_directories(sgemm_v0 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(sgemm_v0 PRIVATE
    CUDA::cublas
)

set_target_properties(sgemm_v0 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
