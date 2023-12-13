// MP 1
#include "wb.h"

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if ( i < len) out[i] = in1[i] + in2[i];

}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  size_t the_lenth = inputLength * sizeof(float);
  cudaMalloc((void**)&deviceInput1, the_lenth);
  cudaMalloc((void**)&deviceInput2, the_lenth);
  cudaMalloc((void**)&deviceOutput, the_lenth);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceInput1,hostInput1,the_lenth,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2,hostInput2,the_lenth,cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");
  dim3 DimGrid(ceil(inputLength/256.0), 1, 1);

  dim3 DimBlok(256, 1, 1);
  //@@ Initialize the grid and block dimensions here

  wbTime_start(Compute, "Performing CUDA computation");
  vecAdd<<<DimGrid,DimBlok>>>(deviceInput1,deviceInput2,deviceOutput,inputLength);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");
  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostOutput,deviceOutput,the_lenth,cudaMemcpyDeviceToHost);


  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);


  wbTime_stop(GPU, "Freeing GPU Memory");
  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
