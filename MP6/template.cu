// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void add(float *input, float *output, float *sum, int len) {
  int index =  (blockIdx.x * blockDim.x * 2);
  int tx = threadIdx.x;
  __shared__ float partial_sum;
  if (tx == 0 ){
    if(blockIdx.x !=0)    partial_sum =  sum[blockIdx.x - 1];         //partial sum for block 1 is the sum in [0]
    else partial_sum = 0;
  }  
  __syncthreads();

  if (index + 2 * tx < len)
    output[index+2 * tx] = input[index+2 * tx] + partial_sum;
  if (index + 2 * tx + 1 < len)
    output[index + 2 * tx + 1 ] = input[index + 2 * tx + 1] + partial_sum;
}

__global__ void scan(float *input, float *output, int len, int mode) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float the_tree[BLOCK_SIZE*2]; 
  int tx = threadIdx.x;
  int offset;
  if(mode==0){
    offset = (blockDim.x*blockIdx.x*2);
    if(tx*2 + offset < len){
      the_tree[tx*2] = input[tx*2+offset];
    }else{
      the_tree[tx*2] = 0;
    }
    if(tx*2 + 1 + offset < len){
      the_tree[tx*2+1] = input[tx*2 + offset +1];
    }else{
      the_tree[tx*2+1] = 0;
    }
  }else{
    offset = (blockDim.x*2);
    if((tx*2+1)*offset-1<len){
      the_tree[tx*2] = input[(tx*2+1)*offset-1];
    }else{
      the_tree[tx*2] = 0;
    }
    if((tx*2+2)*offset-1<len){
      the_tree[tx*2+1] = input[(tx*2+2)*offset-1];
    }else{
      the_tree[tx*2+1] = 0;
    }
  }
  __syncthreads();

  int stride = 1;
  while (stride < 2* BLOCK_SIZE){
    __syncthreads();
    int index = (tx+1)*stride*2-1;
    if(index < 2*BLOCK_SIZE && index >= stride){
      the_tree[index] += the_tree[index - stride];
    }
    stride*=2;
  }
  stride = BLOCK_SIZE/2;
  while (stride >0){
    __syncthreads();
    int index = (tx+1)*stride*2-1;
    if((index+stride)<BLOCK_SIZE*2){
      the_tree[index + stride] +=the_tree[index];
    }
    stride/=2;
  }
  __syncthreads();
  if(mode==1) offset =0;
  if(tx * 2 + offset < len){
    output[tx*2+offset]=the_tree[tx*2];
  }else{
    output[tx*2+offset] = 0;
  }
  if(tx*2 + 1 + offset < len){
    output[tx*2+offset+1]=the_tree[tx*2+1];
  }else{
    output[tx*2+offset+1] = 0;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *devicebuffer;
  float *devicestoresum;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&devicebuffer, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&devicestoresum, 2 * BLOCK_SIZE * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numElements/(2.0*BLOCK_SIZE)), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid,dimBlock>>>(deviceInput, devicebuffer, numElements , 0);
  cudaDeviceSynchronize();

  dim3 bufferGrid(1, 1, 1);
  scan<<<bufferGrid, dimBlock>>>(devicebuffer, devicestoresum, numElements, 1);
  cudaDeviceSynchronize();

  add<<<dimGrid, dimBlock>>>(devicebuffer, deviceOutput, devicestoresum, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
