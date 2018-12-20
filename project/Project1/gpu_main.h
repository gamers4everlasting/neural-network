#ifndef GPULib
#define GPULib

#include <cuda.h>
#include <curand.h>                 // includes random num stuff
#include <curand_kernel.h>          // more rand stuff
#include <cuda_texture_types.h>
#include <stdio.h>

#define cH2D            cudaMemcpyHostToDevice
#define cD2D            cudaMemcpyDeviceToDevice
#define cD2H            cudaMemcpyDeviceToHost

struct Visual_Pkg;

struct GPU_Palette{

    unsigned int palette_width;
    unsigned int palette_height;
    unsigned long num_pixels;

    dim3 gThreads;
    dim3 gBlocks;


    float* red;
    float* green;
    float* blue;
    //float* dft;

};

GPU_Palette initGPUPalette(unsigned int, unsigned int);
int updatePalette(GPU_Palette* P,Visual_Pkg* package, int packetN);
int flushPalette(GPU_Palette*);
void freeGPUPalette(GPU_Palette*);

// kernel calls:

__global__ void updateReds(float* red, unsigned long, double**);
__global__ void updateGreens(float* green, unsigned long, double**);
__global__ void updateBlues(float* blue, unsigned long, double**);
__global__ void flushReds(float* red, unsigned long);
__global__ void flushGreens(float* green, unsigned long);
__global__ void flushBlues(float* blue, unsigned long);

#endif  // GPULib
