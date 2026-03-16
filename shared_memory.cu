#include <cuda_runtime.h>
#include <stdio.h>

__global__ void withoutSharedMemory(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float sum = 0.0f;
        for (int i = -2; i <= 2; i++) {
            int pos = idx + i;
            if (pos >= 0 && pos < n) {
                sum += input[pos];
            }
        }
        output[idx] = sum / 5.0f;
    }
}

__global__ void withSharedMemory(float* input, float* output, int n) {
    extern __shared__ float shared[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    shared[tid + 2] = (idx < n) ? input[idx] : 0.0f;
    
    if (tid < 2) {
        shared[tid] = (idx - 2 >= 0 && idx - 2 < n) ? input[idx - 2] : 0.0f;
        shared[blockDim.x + 2 + tid] = (idx + blockDim.x < n) ? input[idx + blockDim.x] : 0.0f;
    }
    
    __syncthreads();
    
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < 5; i++) {
            sum += shared[tid + i];
        }
        output[idx] = sum / 5.0f;
    }
}

__global__ void matrixMultiplyShared(float* A, float* B, float* C, int N) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * 32 + ty;
    int col = bx * 32 + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (N + 31) / 32; tile++) {
        if (row < N && tile * 32 + tx < N) {
            As[ty][tx] = A[row * N + tile * 32 + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && tile * 32 + ty < N) {
            Bs[ty][tx] = B[(tile * 32 + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < 32; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" {
    void benchmarkSharedMemory(int n) {
        float* h_input = (float*)malloc(n * sizeof(float));
        float* h_output = (float*)malloc(n * sizeof(float));
        
        for (int i = 0; i < n; i++) {
            h_input[i] = (float)i;
        }
        
        float *d_input, *d_output;
        cudaMalloc(&d_input, n * sizeof(float));
        cudaMalloc(&d_output, n * sizeof(float));
        
        cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
        
        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        withoutSharedMemory<<<numBlocks, blockSize>>>(d_input, d_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("Without Shared Memory (1D stencil): %.3f ms\n", milliseconds);
        
        int sharedMemSize = (blockSize + 4) * sizeof(float);
        
        cudaEventRecord(start);
        withSharedMemory<<<numBlocks, blockSize, sharedMemSize>>>(d_input, d_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("With Shared Memory (1D stencil): %.3f ms\n", milliseconds);
        
        int matrixSize = 1024;
        size_t matrixBytes = matrixSize * matrixSize * sizeof(float);
        
        float *h_A = (float*)malloc(matrixBytes);
        float *h_B = (float*)malloc(matrixBytes);
        float *h_C = (float*)malloc(matrixBytes);
        
        for (int i = 0; i < matrixSize * matrixSize; i++) {
            h_A[i] = 1.0f;
            h_B[i] = 1.0f;
        }
        
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, matrixBytes);
        cudaMalloc(&d_B, matrixBytes);
        cudaMalloc(&d_C, matrixBytes);
        
        cudaMemcpy(d_A, h_A, matrixBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, matrixBytes, cudaMemcpyHostToDevice);
        
        dim3 block(32, 32);
        dim3 grid((matrixSize + 31) / 32, (matrixSize + 31) / 32);
        
        cudaEventRecord(start);
        matrixMultiplyShared<<<grid, block>>>(d_A, d_B, d_C, matrixSize);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        double gflops = (2.0 * matrixSize * matrixSize * matrixSize) / (milliseconds * 1e6);
        
        printf("Matrix Multiplication with Shared Memory:\n");
        printf("  Time: %.3f ms\n", milliseconds);
        printf("  Performance: %.2f GFLOPS\n", gflops);
        
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_input);
        free(h_output);
        free(h_A);
        free(h_B);
        free(h_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}
