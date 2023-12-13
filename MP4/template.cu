#include "wb.h"

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define Kernel_width 3
#define Tile_width 4
#define Radius 1
__constant__ float conluvtion_kernel[Kernel_width][Kernel_width][Kernel_width];

//@@ Define constant memory for device kernel here

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__   float N_tile[Tile_width+2*Radius][Tile_width+2*Radius][Tile_width+2*Radius];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int row_out = Tile_width*blockIdx.y + ty;
  int col_out = Tile_width*blockIdx.x + tx;
  int straight_out = Tile_width*blockIdx.z + tz;
  int row_in = row_out -Radius;
  int col_in = col_out - Radius;
  int straight_in = straight_out - Radius;


  float P_temp = 0 ;
  if(row_in >= 0 && col_in >= 0 && straight_in >= 0 && row_in < y_size && col_in < x_size && straight_in < z_size ){
    N_tile[tx][ty][tz] = input[row_in*x_size+col_in+straight_in*x_size*y_size];
  }else{
    N_tile[tx][ty][tz] = 0;
  }
  __syncthreads();
  if(tx<Tile_width && ty< Tile_width && tz < Tile_width){
    for(int i = 0 ; i < Kernel_width; i++){
      for(int j =0 ; j < Kernel_width ; j++){
        for(int k = 0 ; k < Kernel_width ; k++){
          P_temp += conluvtion_kernel[i][j][k] * N_tile[tx+i][ty+j][tz+k];
        }
      }
    }
    __syncthreads();
    if(row_out<y_size && col_out<x_size && straight_out<z_size) output[row_out*x_size+col_out+straight_out*x_size*y_size]=P_temp;
  }

}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);     //the first three element represent the dimension, they are not data 
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int the_size = (inputLength-3)*sizeof(float);
  cudaMalloc((void**)&deviceInput,the_size);
  cudaMalloc((void**)&deviceOutput,the_size);  //for common convolution, the input size and output size is the same 
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  float* starting_address = hostInput+3;
  cudaMemcpy(deviceInput,starting_address,the_size,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(conluvtion_kernel,hostKernel,kernelLength*sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 Dimgrid(ceil((1.0*x_size)/Tile_width),ceil((1.0*y_size)/Tile_width),ceil((1.0*z_size)/Tile_width));       //按理来说矩阵是先行后列，先row再column
  dim3 DimBlock(Tile_width+2*Radius,Tile_width+2*Radius,Tile_width+2*Radius);
  //@@ Launch the GPU kernel here
  conv3d<<<Dimgrid,DimBlock>>>(deviceInput,deviceOutput,z_size,y_size,x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  cudaMemcpy(&hostOutput[3],deviceOutput,the_size,cudaMemcpyDeviceToHost);

  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
