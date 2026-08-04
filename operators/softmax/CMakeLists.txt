add_executable(softmax_v0
    main.cu
    softmax_v0.cu
    softmax_cpu.cpp
    softmax_check.cpp
)

target_include_directories(softmax_v0 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(softmax_v0 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)