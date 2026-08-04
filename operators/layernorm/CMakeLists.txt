add_executable(layernorm_v0
    main.cu
    layernorm_v0.cu
    layernorm_cpu.cpp
    layernorm_check.cpp
)

target_include_directories(layernorm_v0 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(layernorm_v0 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)