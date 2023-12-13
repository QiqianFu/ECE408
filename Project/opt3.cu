#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define Tile_widht 16
__constant__ float kernel_mask[10000];
__global__ void conv_forward_kernel(float* __restrict__ output, const float* __restrict__ input, const float* __restrict__ mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = ((H - K )/S + 1);
    const int W_out = ((W - K )/S + 1);

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) kernel_mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int total_pos = blockIdx.z;
    //each block has the exact number of threads as input image
    int blocks_on_side = (W_out - 1)/ Tile_widht  + 1;
    int W_index = (total_pos % blocks_on_side)* Tile_widht + tx;               //which patch we are at, and which exact pos we are about to handle
    int H_index = (total_pos / blocks_on_side)* Tile_widht + ty;
    int batch_index= blockIdx.x;
    int out_channel_index = blockIdx.y;


    if(W_index < W_out && H_index < H_out){
        float temp = 0 ;
        for(int i = 0; i < C; i++){
            for(int j = 0; j < K; j++){                    
                for (int k = 0; k < K; k += 2) {
                    temp += in_4d(batch_index, i, H_index * S + j, W_index * S + k) * mask_4d(out_channel_index, i, j, k);
                    if (k + 1 < K) {
                        temp += in_4d(batch_index, i, H_index * S + j, W_index * S + k + 1) * mask_4d(out_channel_index, i, j, k + 1);
                    }
            }
        }
        out_4d(batch_index, out_channel_index, H_index, W_index) = temp;
    }








    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    cudaMalloc((void**)device_input_ptr, sizeof(float) * B * C * H * W);
    cudaMalloc((void**)device_output_ptr,sizeof(float) * B * M * ((H - K)/S + 1) * ((W - K)/S + 1));
    // cudaMalloc((void**)device_mask_ptr, sizeof(float) * M * C * K * K);
    cudaMemcpy(*device_input_ptr,host_input,sizeof(float) * B * C * H * W, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,host_mask,sizeof(float) * M * C * K * K, cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(kernel_mask,host_mask,sizeof(float) * M * C * K * K);
    cudaMemcpyToSymbol(kernel_mask, host_mask, sizeof(float) * M * C * K * K);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    int gridW = (((W - K )/S + 1)-1)/Tile_widht+1;
    int gridH = (((H - K )/S + 1)-1)/Tile_widht+1;
    dim3 Dimgrid(B, M, gridW * gridH);    
    dim3 Dimblock( Tile_widht, Tile_widht, 1);
    conv_forward_kernel<<<Dimgrid,Dimblock>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    cudaMemcpy(host_output,device_output,sizeof(float) * B * M * ((H - K)/S + 1) * ((W - K)/S + 1), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}