add_executable(elementwise_v0
    main.cu
    elementwise_v0.cu
    elementwise_check.cpp
)

target_include_directories(elementwise_v0 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(elementwise_v0 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)