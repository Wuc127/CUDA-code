add_executable(reduce_v0
    main.cu
    reduce_v0.cu
    reduce_cpu.cpp
    reduce_check.cpp
)

target_include_directories(reduce_v0 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(reduce_v0 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)