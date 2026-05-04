#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Error check
void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main() {
    const int M = 1024, K = 1024, N = 1024;
    
    // matrix data in host memory
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C = (float*)malloc(M * N * sizeof(float));

    //initialize data
    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // matrix data in device memory
    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, M * K * sizeof(float)));
    checkCuda(cudaMalloc(&d_B, K * N * sizeof(float)));
    checkCuda(cudaMalloc(&d_C, M * N * sizeof(float)));

    // copy the data from host to device
    checkCuda(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    // cuBLAS is column major
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);

    // move the result from host to device
    checkCuda(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Result Matrix C (2x2):" << std::endl;
    for(int i=0; i<M; i++) {
        for(int j=0; j<N; j++) std::cout << h_C[i * N + j] << " ";
        std::cout << std::endl;
    }

    // destroy cuBLAS handle and free the GPU memory
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B), free(h_C);
    
    return 0;
}