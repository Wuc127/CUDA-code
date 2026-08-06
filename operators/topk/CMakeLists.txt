add_executable(topk_v0
    main.cu
    topk_v0.cu
    topk_cpu.cpp
    topk_check.cpp
)

target_include_directories(topk_v0 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(softmax_v0 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)