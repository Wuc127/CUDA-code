add_executable(sgemm_v0
    main.cu
    sgemm_v0.cu
    sgemm_cpu.cpp
    sgemm_check.cpp
)

target_include_directories(sgemm_v0 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(sgemm_v0 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)