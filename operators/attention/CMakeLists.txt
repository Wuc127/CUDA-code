add_executable(attention_v0
    main.cu
    attention_v0.cu
    attention_cpu.cpp
    check_result.cpp
)

target_include_directories(attention_v0 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(attention_v0 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)