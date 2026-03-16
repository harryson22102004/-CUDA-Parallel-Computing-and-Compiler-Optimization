#include <cuda_runtime.h>
#include <stdio.h>

__global__ void reductionNaive(float* input, float* output, int n) {
    extern __shared__ float shared[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    shared[tid] = (idx < n) ? input[idx] : 0.0f;
    
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

__global__ void reductionOptimized(float* input, float* output, int n) {
    extern __shared__ float shared[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float sum = 0.0f;
    
    if (idx < n) {
        sum = input[idx];
    }
    
    if (idx + blockDim.x < n) {
        sum += input[idx + blockDim.x];
    }
    
    shared[tid] = sum;
    
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

__global__ void reductionWarpLevel(float* input, float* output, int n) {
    extern __shared__ float shared[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float sum = 0.0f;
    
    if (idx < n) {
        sum = input[idx];
    }
    
    if (idx + blockDim.x < n) {
        sum += input[idx + blockDim.x];
    }
    
    shared[tid] = sum;
    
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        volatile float* vshared = shared;
        
        if (blockDim.x > 32) vshared[tid] += vshared[tid + 32];
        if (blockDim.x > 16) vshared[tid] += vshared[tid + 16];
        if (blockDim.x > 8) vshared[tid] += vshared[tid + 8];
        if (blockDim.x > 4) vshared[tid] += vshared[tid + 4];
        if (blockDim.x > 2) vshared[tid] += vshared[tid + 2];
        if (blockDim.x > 1) vshared[tid] += vshared[tid + 1];
    }
    
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

extern "C" {
    float launchReduction(float* h_input, int n) {
        float *d_input, *d_output;
        cudaMalloc(&d_input, n * sizeof(float));
        
        int blockSize = 256;
        int numBlocks = (n + blockSize * 2 - 1) / (blockSize * 2);
        
        cudaMalloc(&d_output, numBlocks * sizeof(float));
        
        cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        
        reductionNaive<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(
            d_input, d_output, n);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        float* h_output = new float[numBlocks];
        cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
        
        float finalSum = 0.0f;
        for (int i = 0; i < numBlocks; i++) {
            finalSum += h_output[i];
        }
        
        printf("Reduction (%d elements):\n", n);
        printf("  Time: %.3f ms\n", milliseconds);
        printf("  Bandwidth: %.2f GB/s\n", 
               (n * sizeof(float)) / (milliseconds * 1e6));
        printf("  Result: %f\n", finalSum);
        
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return finalSum;
    }
}
