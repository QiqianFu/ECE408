#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define Tile_widht 32
#define Block_size 1024
__constant__ float kernel_mask[10000];
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, int C, int K , int M, int H, int W, int B)
{
    __shared__ float tile_mask[Tile_widht*Tile_widht];
    __shared__ float tile_input[Tile_widht*Tile_widht];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int index_x = blockIdx.x * blockDim.x + tx;
    int index_y = blockIdx.y * blockDim.y + ty;
    int tz = blockIdx.z;
    float temp = 0;
    int index ;
    for(int i = 0 ; i < ((C * K * K -1)/Tile_widht) + 1; i++ ){
        index = i * Tile_widht + ty;
        if ( index_x < H*W && index < C * K * K ){
            tile_input[ty * Tile_widht+tx] = input[C * K * K  * H * W * tz + index * H * W + index_x];
        }else{
            tile_input[ty * Tile_widht+tx] = 0;            
        }  
        index = i * Tile_widht + tx;
        if ( index_y < M && index < C * K * K ){
            tile_mask[ty * Tile_widht+tx] = kernel_mask[index_y * C * K * K + index];
        }else{
            tile_mask[ty * Tile_widht+tx] = 0;            
        }      
        __syncthreads();
        if ((index_y < M) && ( index_x < H*W )){
            for(int i =0 ; i < Tile_widht; i++){
                temp += tile_mask[ty*Tile_widht+i] * tile_input[ i *Tile_widht+tx]; 
            }
            output[tz*H*W*M+index_y*H*W+index_x] = temp;
        }        
        __syncthreads();
    }
}

__global__ void unfold(float *device_input_ptr, float* device_input_rearrange, int C , int K , int H ,int W ,int S){
    int tx = (blockDim.x * blockIdx.x + threadIdx.x );
    int gridW = ((W - K)/S + 1);
    int gridH = ((H - K)/S + 1);
    int batch = blockIdx.z;
    int x = K * K;
    int y = gridW * gridH;
    int z = C;
    int current_channel;
    int current_index;
    int current_h;
    int current_w ;
    if (tx < y ){
        current_channel = blockIdx.y;
        current_index = tx ;
        current_h = current_index / gridW;
        current_w = current_index % gridW; 
        for(int i = 0 ; i < x ; i++){
            device_input_ptr[batch * x * y * z  +  (i+current_channel*x)*y + current_h * gridW + current_w] = device_input_rearrange[batch * C * H * W + current_channel* H * W + (current_h * S +i/K) * W + current_w * S + i%K];
        }
    }

}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    int gridW = ((W - K )/S + 1);
    int gridH = ((H - K )/S + 1);
    cudaMalloc((void**)device_input_ptr,sizeof(float)* C * K * K * gridH * gridW * B);
    float *device_input_rearrange;
    cudaMalloc((void**)&device_input_rearrange, sizeof(float) * B * C * H * W);
    cudaMalloc((void**)device_output_ptr,sizeof(float) * B * M * ((H - K)/S + 1) * ((W - K)/S + 1));
    cudaMemcpy(device_input_rearrange,host_input,sizeof(float) * B * C * H * W, cudaMemcpyHostToDevice);
    dim3 Dimgrid(ceil(gridW * gridH / (1.0 * Block_size)), C , B);
    dim3 Dimblock(Block_size,1,1);
    unfold<<<Dimgrid, Dimblock >>>(*device_input_ptr , device_input_rearrange ,C,K,H,W,S);
    cudaDeviceSynchronize();
    cudaMemcpy(*device_mask_ptr,host_mask,sizeof(float) * M * C * K * K, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel_mask, host_mask, sizeof(float) * M * C * K * K);
    cudaFree(device_input_rearrange);


}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    int gridW = ((W - K )/S + 1);
    int gridH = ((H - K )/S + 1);
    dim3 Dimgrid (( gridW * gridH - 1) / Tile_widht + 1,  (M - 1) / Tile_widht + 1, B);
    dim3 Dimblock (Tile_widht, Tile_widht, 1);
    conv_forward_kernel<<<Dimgrid,Dimblock>>>(device_output, device_input, device_mask, C , K, M , gridH, gridW, B);

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
