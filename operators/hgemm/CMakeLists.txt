add_executable(hgemm_v0
    main.cu
    hgemm_v0.cu
    hgemm_cpu.cpp
    hgemm_check.cpp
)

target_include_directories(hgemm_v0 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(hgemm_v0 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)