/*******************************************************************************
*
*   Music visualizer base code by Dr. Michael Brady
*
*******************************************************************************/

#include "interface.h"

int VERBOSE = 1; 		// only used for interface
int RUNMODE = 1;
volatile int packet_pos = 0;
volatile int print_spectrum = 0;
static volatile int time_to_exit = 0;



/******************************************************************************/
int main(int argc, char *argv[]){

	unsigned char ch;


	ch = crack(argc, argv, nullptr, 0);


	char* fileName1 = argv[arg_index];

	GPU_Palette P1;
	P1 = openPalette(1024, 512); // width, height
	CPUAnimBitmap animation(&P1);
	cudaMalloc((void**) &animation.dev_bitmap, animation.image_size());
  animation.initAnimation();

	switch(RUNMODE){
			case 0:
				break;
			case 1:
				if (VERBOSE) printf("\n -- running music visualizer -- \n");
				runAudio(fileName1, &P1, &animation);
				break;
			default: printf("no valid run mode selected\n");
	}

	return 0;
}
int usage()
{
	printf("USAGE:\n");
	printf("-r[val] filename\n\n");
  printf("e.g.> ex2 -r1 -v1 filename.wav\n");
  printf("v  verbose mode (0:none, 1:normal, 2:params\n");
  printf("r  run mode 0:GPU info, 1:music visualizer\n\n");
	printf("note: be sure .wav file is 16 bit, 2 channel, 44.1kHz\n");

  return(0);
}

GPU_Palette openPalette(int theWidth, int theHeight)
{
	unsigned long theSize = theWidth * theHeight;

	unsigned long memSize = theSize * sizeof(float);
	float* redmap = (float*) malloc(memSize);
	float* greenmap = (float*) malloc(memSize);
	float* bluemap = (float*) malloc(memSize);

	for(int i = 0; i < theSize; i++)
	{
  	bluemap[i]    = 0;
  	greenmap[i]  = 0;
  	redmap[i]   = 0;
	}

	GPU_Palette P1 = initGPUPalette(theWidth, theHeight);

	//cudaMemcpy(P1.gray, graymap, memSize, cH2D);
	cudaMemcpy(P1.red, redmap, memSize, cH2D);
	cudaMemcpy(P1.green, greenmap, memSize, cH2D);
	cudaMemcpy(P1.blue, bluemap, memSize, cH2D);

	//free(graymap);
	free(redmap);
	free(greenmap);
	free(bluemap);

	return P1;
}
