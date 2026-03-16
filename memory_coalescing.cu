#include <cuda_runtime.h>
#include <stdio.h>

__global__ void uncoalescedAccess(float* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        data[y * width + x] = data[y * width + x] * 2.0f;
    }
}

__global__ void coalescedAccess(float* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        data[y * width + x] = data[y * width + x] * 2.0f;
    }
}

__global__ void transposeNaive(float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
    }
}

__global__ void transposeCoalesced(float* input, float* output, int width, int height) {
    __shared__ float tile[32][32];
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    
    __syncthreads();
    
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

extern "C" {
    void benchmarkMemoryAccess(int width, int height) {
        size_t size = width * height * sizeof(float);
        
        float* h_data = (float*)malloc(size);
        float* h_result = (float*)malloc(size);
        
        for (int i = 0; i < width * height; i++) {
            h_data[i] = (float)i;
        }
        
        float *d_data, *d_output;
        cudaMalloc(&d_data, size);
        cudaMalloc(&d_output, size);
        
        cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
        
        dim3 block(32, 32);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        uncoalescedAccess<<<grid, block>>>(d_data, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("Uncoalesced Access: %.3f ms\n", milliseconds);
        
        cudaEventRecord(start);
        coalescedAccess<<<grid, block>>>(d_data, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("Coalesced Access: %.3f ms\n", milliseconds);
        
        dim3 transposeGrid((height + 31) / 32, (width + 31) / 32);
        
        cudaEventRecord(start);
        transposeNaive<<<transposeGrid, block>>>(d_data, d_output, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("Transpose Naive: %.3f ms\n", milliseconds);
        
        cudaEventRecord(start);
        transposeCoalesced<<<transposeGrid, block>>>(d_data, d_output, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("Transpose Coalesced: %.3f ms\n", milliseconds);
        
        cudaFree(d_data);
        cudaFree(d_output);
        free(h_data);
        free(h_result);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}
