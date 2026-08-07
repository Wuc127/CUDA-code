add_executable(transpose_v0
    main.cu
    transpose_v0.cu
    transpose_cpu.cpp
    transpose_check.cpp
)

target_include_directories(transpose_v0 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(softmax_v0 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)