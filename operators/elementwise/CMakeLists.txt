add_executable(elementwise
    main.cu
    elementwise_v0.cu
    elementwise_v1.cu
    elementwise_v2.cu
    elementwise_v3.cu
    elementwise_check.cpp
)

target_include_directories(elementwise PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(elementwise PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)