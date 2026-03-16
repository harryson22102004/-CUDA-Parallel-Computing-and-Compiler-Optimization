#include <cuda_runtime.h>
#include <stdio.h>

__global__ void convolutionNaive(float* input, float* kernel, float* output, 
                                  int inputHeight, int inputWidth, 
                                  int kernelHeight, int kernelWidth) {
    int outH = blockIdx.y * blockDim.y + threadIdx.y;
    int outW = blockIdx.x * blockDim.x + threadIdx.x;
    
    int outputHeight = inputHeight - kernelHeight + 1;
    int outputWidth = inputWidth - kernelWidth + 1;
    
    if (outH < outputHeight && outW < outputWidth) {
        float sum = 0.0f;
        
        for (int kh = 0; kh < kernelHeight; kh++) {
            for (int kw = 0; kw < kernelWidth; kw++) {
                int inH = outH + kh;
                int inW = outW + kw;
                sum += input[inH * inputWidth + inW] * kernel[kh * kernelWidth + kw];
            }
        }
        
        output[outH * outputWidth + outW] = sum;
    }
}

__global__ void convolutionSharedMemory(float* input, float* kernel, float* output,
                                         int inputHeight, int inputWidth,
                                         int kernelHeight, int kernelWidth) {
    extern __shared__ float sharedInput[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int outH = blockIdx.y * blockDim.y + ty;
    int outW = blockIdx.x * blockDim.x + tx;
    
    int outputHeight = inputHeight - kernelHeight + 1;
    int outputWidth = inputWidth - kernelWidth + 1;
    
    int sharedHeight = blockDim.y + kernelHeight - 1;
    int sharedWidth = blockDim.x + kernelWidth - 1;
    
    for (int i = ty; i < sharedHeight; i += blockDim.y) {
        for (int j = tx; j < sharedWidth; j += blockDim.x) {
            int globalH = blockIdx.y * blockDim.y + i;
            int globalW = blockIdx.x * blockDim.x + j;
            
            if (globalH < inputHeight && globalW < inputWidth) {
                sharedInput[i * sharedWidth + j] = input[globalH * inputWidth + globalW];
            } else {
                sharedInput[i * sharedWidth + j] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    if (outH < outputHeight && outW < outputWidth) {
        float sum = 0.0f;
        
        for (int kh = 0; kh < kernelHeight; kh++) {
            for (int kw = 0; kw < kernelWidth; kw++) {
                sum += sharedInput[(ty + kh) * sharedWidth + (tx + kw)] * 
                       kernel[kh * kernelWidth + kw];
            }
        }
        
        output[outH * outputWidth + outW] = sum;
    }
}

extern "C" {
    void launchConvolution(float* h_input, float* h_kernel, float* h_output,
                          int inputHeight, int inputWidth,
                          int kernelHeight, int kernelWidth) {
        float *d_input, *d_kernel, *d_output;
        size_t inputBytes = inputHeight * inputWidth * sizeof(float);
        size_t kernelBytes = kernelHeight * kernelWidth * sizeof(float);
        int outputHeight = inputHeight - kernelHeight + 1;
        int outputWidth = inputWidth - kernelWidth + 1;
        size_t outputBytes = outputHeight * outputWidth * sizeof(float);
        
        cudaMalloc(&d_input, inputBytes);
        cudaMalloc(&d_kernel, kernelBytes);
        cudaMalloc(&d_output, outputBytes);
        
        cudaMemcpy(d_input, h_input, inputBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, h_kernel, kernelBytes, cudaMemcpyHostToDevice);
        
        dim3 block(16, 16);
        dim3 grid((outputWidth + block.x - 1) / block.x,
                  (outputHeight + block.y - 1) / block.y);
        
        size_t sharedMemSize = (block.y + kernelHeight - 1) * 
                               (block.x + kernelWidth - 1) * sizeof(float);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        convolutionNaive<<<grid, block>>>(d_input, d_kernel, d_output,
                                          inputHeight, inputWidth,
                                          kernelHeight, kernelWidth);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        convolutionSharedMemory<<<grid, block, sharedMemSize>>>(
            d_input, d_kernel, d_output,
            inputHeight, inputWidth,
            kernelHeight, kernelWidth);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaMemcpy(h_output, d_output, outputBytes, cudaMemcpyDeviceToHost);
        
        printf("Convolution (%dx%d input, %dx%d kernel):\n", 
               inputHeight, inputWidth, kernelHeight, kernelWidth);
        printf("  Time: %.3f ms\n", milliseconds);
        printf("  Operations: %.2f GFLOPs\n", 
               (2.0 * outputHeight * outputWidth * kernelHeight * kernelWidth) / (milliseconds * 1e6));
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_input);
        cudaFree(d_kernel);
        cudaFree(d_output);
    }
}
