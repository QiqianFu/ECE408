
#include <wb.h>
#define Tile_Width 16
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subTileM[256];
  __shared__ float subTileN[256];
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  //now begin to load tiles into share memory
  float P_temp = 0;
  for(int i =0; i < ceil((1.0*numAColumns)/16) ; i++){     //one output tile may need many tiles from A and B 
    if( threadIdx.x+i*Tile_Width < numAColumns && row < numARows){
      subTileM[threadIdx.y*Tile_Width+threadIdx.x]=A[threadIdx.x+i*Tile_Width+row*numAColumns];
    }else{
      subTileM[threadIdx.y*Tile_Width+threadIdx.x]=0;
    }
    if( threadIdx.y+i*Tile_Width < numAColumns && col < numBColumns){
      subTileN[threadIdx.y*Tile_Width+threadIdx.x]=B[(threadIdx.y+i*Tile_Width)*numBColumns+col];
    }else{
      subTileN[threadIdx.y*Tile_Width+threadIdx.x]=0;
    }
    __syncthreads();

    for(int j =0 ; j < Tile_Width; j++){
      P_temp += subTileM[threadIdx.y*Tile_Width+j]*subTileN[threadIdx.x+j*Tile_Width];
    }
    __syncthreads();
  if(row<numCRows && col < numCColumns)  C[col+row*numCColumns] = P_temp;
  }


}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  size_t sizeA = numAColumns*numARows*sizeof(float);
  size_t sizeB = numBColumns*numBRows*sizeof(float);
  size_t sizeC = numCColumns*numCRows*sizeof(float);
  hostC = (float*)malloc(sizeC);

  //@@ Allocate the hostC matrix
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceC,sizeC);
  cudaMalloc((void**)&deviceB,sizeB);
  cudaMalloc((void**)&deviceA,sizeA);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA,hostA,sizeA,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB,hostB,sizeB,cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  
  dim3 Dimgrid(ceil((1.0*numCColumns)/16.0),ceil((1.0*numCRows)/16.0),1);       //按理来说矩阵是先行后列，先row再column
  dim3 DimBlock(16,16,1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<Dimgrid,DimBlock>>>(deviceA,deviceB,deviceC,numARows,numAColumns,numBRows,numBColumns, numCRows,numCColumns);


  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC,deviceC,sizeC,cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
