/**************************************************************************
*
*     Music Visualizer Base Code
*
**************************************************************************/

// define texture memory
// texture<float, 2> texGray;
texture<float, 2> texBlue;
texture<float, 2> texGreen;
texture<float, 2> texRed;

//unsigned long num_pixels;
#include "gpu_main.h"

/******************************************************************************/
GPU_Palette initGPUPalette(unsigned int imageWidth, unsigned int imageHeight)
{
  GPU_Palette X;

  X.gThreads.x = 32;  // 32 x 32 = 1024 threads per block
  X.gThreads.y = 32;
  X.gThreads.z = 1;
  X.gBlocks.x = ceil(imageWidth/32);  // however many blocks needed for image
  X.gBlocks.y = ceil(imageWidth/32);
  X.gBlocks.z = 1;

  X.palette_width = imageWidth;       // save this info
  X.palette_height = imageHeight;
  X.num_pixels = imageWidth * imageHeight;

  // allocate memory on GPU
  cudaError_t err;
  err = cudaMalloc((void**) &X.red, X.num_pixels * sizeof(float));
  if(err != cudaSuccess){
    printf("cuda error allocating red = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  cudaMalloc((void**) &X.green, X.num_pixels * sizeof(float)); // g
  if(err != cudaSuccess){
    printf("cuda error allocating green = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  cudaMalloc((void**) &X.blue, X.num_pixels * sizeof(float));  // b
  if(err != cudaSuccess){
    printf("cuda error allocating blue = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  //cudaMalloc((void**) &X.rand, X.num_pixels * sizeof(curandState));
  if(err != cudaSuccess){
    printf("cuda error allocating rands = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  // dft size = 1024
  //cudaMalloc((void**) &X.dft, 1024 * sizeof(float));
  if(err != cudaSuccess){
    printf("cuda error allocating dft = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }


  // init rand seeds on the gpu
  //setup_rands <<< X.gBlocks, X.gThreads >>> (X.rand, time(NULL), X.num_pixels);

  // create texture memory and bind
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  unsigned int pitch = sizeof(float) * imageWidth;
  cudaBindTexture2D(NULL, texBlue, X.blue, desc, imageWidth, imageHeight, pitch);
  cudaBindTexture2D(NULL, texRed, X.red, desc, imageWidth, imageHeight, pitch);
  cudaBindTexture2D(NULL, texGreen, X.green, desc, imageWidth, imageHeight, pitch);
  return X;
}

/******************************************************************************/
void freeGPUPalette(GPU_Palette* P)
{
  // free texture memory
  cudaUnbindTexture(texBlue); // this is bound to black and white
  cudaUnbindTexture(texGreen);
  cudaUnbindTexture(texRed);
  // free gpu memory
//  cudaFree(P->gray);
  cudaFree(P->red);
  cudaFree(P->green);
  cudaFree(P->blue);
  //cudaFree(P->dft);
  //cudaFree(P->rand);
}
int flushPalette(GPU_Palette* P)
{
  flushReds <<< P->gBlocks, P->gThreads >>> (P->red, P->num_pixels);
  flushGreens <<< P->gBlocks, P->gThreads >>> (P->green, P->num_pixels);
	flushBlues <<< P->gBlocks, P->gThreads >>> (P->blue, P->num_pixels);
  return 0;
}
__global__ void flushReds(float* red, unsigned long numPixels)
{
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  if(vecIdx < numPixels){
    if(y >= 7 && y <= 57) red[vecIdx] = 0.5;
    else if(y >= 63 && y <= 113) red[vecIdx] = 0.5;
    else if(y >= 119 && y <= 169) red[vecIdx] = 0.5;
    else if(y >= 175 && y <= 225) red[vecIdx] = 0.5;
    else if(y >= 231 && y <= 281) red[vecIdx] = 0.5;
    else if(y >= 287 && y <= 337) red[vecIdx] = 0.5;
    else if(y >= 343 && y <= 393) red[vecIdx] = 0.5;
    else if(y >= 399 && y <= 449) red[vecIdx] = 0.5;
    else if(y >= 455 && y <= 505) red[vecIdx] = 0.5;
  }
}
__global__ void flushGreens(float* green, unsigned long numPixels)
{
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  if(vecIdx < numPixels){
    if(y >= 7 && y <= 57) green[vecIdx] = 0.5;
    else if(y >= 63 && y <= 113) green[vecIdx] = 0.5;
    else if(y >= 119 && y <= 169) green[vecIdx] = 0.5;
    else if(y >= 175 && y <= 225) green[vecIdx] = 0.5;
    else if(y >= 231 && y <= 281) green[vecIdx] = 0.5;
    else if(y >= 287 && y <= 337) green[vecIdx] = 0.5;
    else if(y >= 343 && y <= 393) green[vecIdx] = 0.5;
    else if(y >= 399 && y <= 449) green[vecIdx] = 0.5;
    else if(y >= 455 && y <= 505) green[vecIdx] = 0.5;
  }
}
__global__ void flushBlues(float* blue, unsigned long numPixels)
{
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  if(vecIdx < numPixels){
    if(y >= 7 && y <= 57) blue[vecIdx] = 0.5;
    else if(y >= 63 && y <= 113) blue[vecIdx] = 0.5;
    else if(y >= 119 && y <= 169) blue[vecIdx] = 0.5;
    else if(y >= 175 && y <= 225) blue[vecIdx] = 0.5;
    else if(y >= 231 && y <= 281) blue[vecIdx] = 0.5;
    else if(y >= 287 && y <= 337) blue[vecIdx] = 0.5;
    else if(y >= 343 && y <= 393) blue[vecIdx] = 0.5;
    else if(y >= 399 && y <= 449) blue[vecIdx] = 0.5;
    else if(y >= 455 && y <= 505) blue[vecIdx] = 0.5;
  }
}
/******************************************************************************/
int updatePalette(GPU_Palette* P,Visual_Pkg* package, int packetN)
{
  flushPalette(P);

  // updateReds <<< P->gBlocks, P->gThreads >>> (P->red, P->num_pixels, package);
  // updateGreens <<< P->gBlocks, P->gThreads >>> (P->green, P->num_pixels, package);
	// updateBlues <<< P->gBlocks, P->gThreads >>> (P->blue, P->num_pixels, package);
  //
  // sleep(23000);
  // flushPalette(P);
  return 0;
}

/******************************************************************************/
__global__ void updateReds(float* red, unsigned long numPixels, Visual_Pkg* package)
{
  // assuming 1024w x 512h palette
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);
  //
  // if(vecIdx < numPixels){
  //   if(y >= 7 && y <= 57)
  //   else if(y >= 63 && y <= 113)
  //   else if(y >= 119 && y <= 169)
  //   else if(y >= 175 && y <= 225)
  //   else if(y >= 231 && y <= 281)
  //   else if(y >= 287 && y <= 337)
  //   else if(y >= 343 && y <= 393)
  //   else if(y >= 399 && y <= 449)
  //   else if(y >= 455 && y <= 505)
  // }
}

/******************************************************************************/
__global__ void updateGreens(float* green, unsigned long numPixels, Visual_Pkg* package)
{
  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  // assuming 1024w x 512h palette
  // if(vecIdx < numPixels){ // don't compute pixels out of range
  //   if(y >= 7 && y <= 57)
  //   else if(y >= 63 && y <= 113)
  //   else if(y >= 119 && y <= 169)
  //   else if(y >= 175 && y <= 225)
  //   else if(y >= 231 && y <= 281)
  //   else if(y >= 287 && y <= 337)
  //   else if(y >= 343 && y <= 393)
  //   else if(y >= 399 && y <= 449)
  //   else if(y >= 455 && y <= 505)
  // }
}

/******************************************************************************/
__global__ void updateBlues(float* blue, unsigned long numPixels, Visual_Pkg* package)
{

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int vecIdx = x + (y * blockDim.x * gridDim.x);

  // if(vecIdx < numPixels){
  //   if(y >= 7 && y <= 57)
  //   else if(y >= 63 && y <= 113)
  //   else if(y >= 119 && y <= 169)
  //   else if(y >= 175 && y <= 225)
  //   else if(y >= 231 && y <= 281)
  //   else if(y >= 287 && y <= 337)
  //   else if(y >= 343 && y <= 393)
  //   else if(y >= 399 && y <= 449)
  //   else if(y >= 455 && y <= 505)
  // }
}
