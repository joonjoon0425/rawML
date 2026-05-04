add_executable(test_gemm cuda/test_gemm.cu)

target_link_libraries(test_gemm PRIVATE CUDA::cublas)