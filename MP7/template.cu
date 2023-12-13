// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 128
#define Dim_Block_size 32

//@@ insert code here
__global__ void FromFloat2UnsignedChar(float *input, unsigned char *output, int width, int height){
  int index_width = threadIdx.x + blockDim.x * blockIdx.x;
  int index_height = threadIdx.y + blockDim.y * blockIdx.y ;
  if ( index_width < width && index_height < height){
    output[blockIdx.z * height * width + index_height * width + index_width] = \
    (unsigned char)(255 * input[blockIdx.z * height * width + index_height * width + index_width]);
  }
}

__global__ void FromRGB2GrayScale(unsigned char *input, unsigned char *output, int width, int height){
  int index_width = threadIdx.x + blockDim.x * blockIdx.x;
  int index_height = threadIdx.y + blockDim.y * blockIdx.y ;
  if ( index_width < width && index_height < height){
    int idx = index_height* width + index_width;
    unsigned char r = input[3*idx];
    unsigned char g = input[3*idx+1];
    unsigned char b = input[3*idx+2];
    output[idx] = (unsigned char)(0.21 * r + 0.71 * g + 0.07 * b);
  }
}

__global__ void HistogramofGrayImage(unsigned char *input, unsigned int *output, int width, int height){
  __shared__ unsigned int the_histogram[HISTOGRAM_LENGTH];
  int index_width = threadIdx.x + blockDim.x * blockIdx.x;
  int index_height = threadIdx.y + blockDim.y * blockIdx.y ;
  int idx = threadIdx.x + threadIdx.y * blockDim.x;
  if (idx < HISTOGRAM_LENGTH){
    the_histogram[idx] = 0;
  }
  __syncthreads();
  if (index_width < width && index_height < height){
    int the_pixel = index_height* width + index_width;
    atomicAdd(&(the_histogram[input[the_pixel]]), 1);
  }
  __syncthreads();
  if(idx<HISTOGRAM_LENGTH){
    atomicAdd(&(output[idx]), the_histogram[idx]);
  }
}

// __global__ int atomicCAS(int *address, int old, int new){
//   if (*address != old) return 0;
//   *address = new;
//   return 1;
// }

// __global__ int atomicAdd(int* address, int value) 
// {
//   int done = 0;
//   while (!done) {
//   int old_v = *address;
//   done = atomicCAS(address, old_v, old_v + value);
//   }
//   return old_v;
// }



__global__ void CumulativeDisOfHis(unsigned int *his, float *output, int width, int height){
  __shared__ unsigned int cumulativeHis[HISTOGRAM_LENGTH];
  int tx = threadIdx.x;
  if(tx * 2  < HISTOGRAM_LENGTH){
    cumulativeHis[tx*2] = his[tx*2];
  }
  if(tx * 2 + 1 < HISTOGRAM_LENGTH){
    cumulativeHis[tx*2+1] = his[tx*2+1];
  }
  
  __syncthreads();

  int stride = 1;
  while (stride < 2 * BLOCK_SIZE){
    __syncthreads();
    int index = (tx+1)*stride*2-1;
    if(index < 2*BLOCK_SIZE && index >= stride){
      cumulativeHis[index] += cumulativeHis[index - stride];
    }
    stride*=2;
  }
  stride = BLOCK_SIZE/2;
  while (stride >0){
    __syncthreads();
    int index = (tx+1) * stride * 2-1;
    if((index+stride)<BLOCK_SIZE*2){
      cumulativeHis[index + stride] +=cumulativeHis[index];
    }
    stride/=2;
  }
  __syncthreads();

  if(tx * 2 < HISTOGRAM_LENGTH){
    output[tx*2]=cumulativeHis[tx*2]/(width*height*1.0);
  }
  if(tx*2 + 1 < HISTOGRAM_LENGTH){
    output[tx*2+1]=cumulativeHis[tx*2+1]/(width*height*1.0);
  }
}

__global__ void Equalization(unsigned char* input, float *cdf, int width, int height){
  int index_width = threadIdx.x + blockDim.x * blockIdx.x;
  int index_height = threadIdx.y + blockDim.y * blockIdx.y ;
  int the_index = blockIdx.z * width * height + index_height * width + index_width ;
  if (index_width < width && index_height < height){
    float max_temp = max(255*(cdf[input[the_index]] - cdf[0])/(1.0-cdf[0]), 0.0);
    float min_temp = min(max_temp, 255.0);
    input[the_index] = (unsigned char)min_temp;
  }
}

__global__ void Back2Float(unsigned char *input, float *output, int width, int height){
  int index_width = threadIdx.x + blockDim.x * blockIdx.x;
  int index_height = threadIdx.y + blockDim.y * blockIdx.y ;
  int the_index = blockIdx.z * width * height + index_height * width + index_width ;
  if (index_width < width && index_height < height){
    output[the_index] = (float)(input[the_index]/255.0);
  }
}




int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  float *device_input;
  //@@ Insert more code here

  float *deviceCdf;
  float *device_output;
  unsigned char *deviceUnsignedChar3chans;
  unsigned char *deviceUnsignedChar1chan;
  unsigned int *deviceHistogram;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  cudaMalloc((void**) &device_input,sizeof(float) * imageWidth * imageHeight * imageChannels);
  cudaMalloc((void**) &device_output, sizeof(float) * imageWidth * imageHeight * imageChannels);
  cudaMalloc((void**) &deviceUnsignedChar3chans, sizeof(unsigned char) * imageWidth * imageHeight * imageChannels);
  cudaMalloc((void**) &deviceUnsignedChar1chan, sizeof(unsigned char) * imageWidth * imageHeight);
  cudaMalloc((void**) &deviceHistogram, sizeof(unsigned int) * HISTOGRAM_LENGTH);
  cudaMalloc((void**) &deviceCdf, sizeof(float) * HISTOGRAM_LENGTH);
  cudaMemset((void*) deviceHistogram, 0 , HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset((void*) deviceCdf, 0 , HISTOGRAM_LENGTH * sizeof(float));

  cudaMemcpy(device_input, hostInputImageData,sizeof(float) * imageWidth * imageHeight * imageChannels, cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil( imageWidth / (1.0 * Dim_Block_size)), ceil( imageHeight / (1.0 * Dim_Block_size)), imageChannels);
  dim3 dimBlock(Dim_Block_size, Dim_Block_size, 1);
  FromFloat2UnsignedChar<<<dimGrid, dimBlock>>>(device_input,deviceUnsignedChar3chans, imageWidth , imageHeight);
  
  cudaDeviceSynchronize();

  dimGrid = dim3(ceil( imageWidth / (1.0 * Dim_Block_size)), ceil( imageHeight / (1.0 * Dim_Block_size)), 1);
  dimBlock = dim3(Dim_Block_size, Dim_Block_size, 1);
  FromRGB2GrayScale<<<dimGrid,dimBlock>>>(deviceUnsignedChar3chans , deviceUnsignedChar1chan, imageWidth , imageHeight);

  cudaDeviceSynchronize();

  HistogramofGrayImage<<<dimGrid,dimBlock>>>(deviceUnsignedChar1chan, deviceHistogram, imageWidth , imageHeight);

  cudaDeviceSynchronize();

  dimGrid = dim3(1, 1, 1);
  dimBlock = dim3(BLOCK_SIZE, 1, 1);
  CumulativeDisOfHis<<<dimGrid,dimBlock>>>(deviceHistogram, deviceCdf,imageWidth , imageHeight);
  cudaDeviceSynchronize();

  dimGrid = dim3(ceil( imageWidth / (1.0 * Dim_Block_size)), ceil( imageHeight / (1.0 * Dim_Block_size)), imageChannels);
  dimBlock = dim3(Dim_Block_size, Dim_Block_size, 1);
  Equalization<<<dimGrid, dimBlock>>>(deviceUnsignedChar3chans, deviceCdf, imageWidth , imageHeight);
  cudaDeviceSynchronize();

  Back2Float<<<dimGrid, dimBlock>>>(deviceUnsignedChar3chans ,device_output, imageWidth , imageHeight);
  //@@ insert code here
  cudaDeviceSynchronize();
  cudaMemcpy(hostOutputImageData, device_output, sizeof(float) * imageWidth * imageHeight * imageChannels, cudaMemcpyDeviceToHost);
  wbSolution(args, outputImage);
  cudaFree(device_input);
  cudaFree(device_output);
  cudaFree(deviceUnsignedChar3chans);
  cudaFree(deviceUnsignedChar1chan);
  cudaFree(deviceHistogram);
  cudaFree(deviceCdf);
  free(hostOutputImageData);
  free(hostInputImageData);

  return 0;
}
