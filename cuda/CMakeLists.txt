add_library(cs179_cuda STATIC
    src/kernels.cu
)

target_include_directories(cs179_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(cs179_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(cs179_cuda PUBLIC CUDA::cudart)
