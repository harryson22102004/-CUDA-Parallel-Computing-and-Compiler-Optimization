#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void matrixMulNaive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matrixMulOptimized(float* A, float* B, float* C, int N) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (N + blockDim.x - 1) / blockDim.x; tile++) {
        if (row < N && tile * blockDim.x + tx < N) {
            As[ty][tx] = A[row * N + tile * blockDim.x + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && tile * blockDim.y + ty < N) {
            Bs[ty][tx] = B[(tile * blockDim.y + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < blockDim.x; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#include <mma.h>
using namespace nvcuda;

__global__ void matrixMulTensorCore(float* A, float* B, float* C, int N) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, float, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, float, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    wmma::load_matrix_sync(a_frag, A + blockIdx.x * 16, N);
    wmma::load_matrix_sync(b_frag, B + blockIdx.x * 16 * N, N);
    
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    wmma::store_matrix_sync(C + blockIdx.x * 16, c_frag, N, wmma::mem_row_major);
}
#endif

extern "C" {
    void launchMatrixMultiplication(float* h_A, float* h_B, float* h_C, int N, int blockSize) {
        float *d_A, *d_B, *d_C;
        size_t bytes = N * N * sizeof(float);
        
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);
        
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
        
        dim3 block(blockSize, blockSize);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        matrixMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        matrixMulOptimized<<<grid, block>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
        
        double gflops = (2.0 * N * N * N) / (milliseconds * 1e6);
        printf("Matrix multiplication (%d x %d):\n", N, N);
        printf("  Time: %.3f ms\n", milliseconds);
        printf("  Performance: %.2f GFLOPS\n", gflops);
        printf("  Bandwidth: %.2f GB/s\n", (bytes * 3) / (milliseconds * 1e6));
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}
