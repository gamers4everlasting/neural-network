#ifndef AUDIOLib
#define AUDIOLib

#include <complex.h>
#include <fftw3.h>
#include <stdio.h>
#include <SDL2/SDL.h> // ans Uint8, etc..
#include <math.h>
//#include "gpu_main.h"
#include "animate.h"

enum {re, im};
struct AudioData
{
  Uint8* currentPos;
	Uint32 currentLength;
	Uint8* wavPtr;
  Uint32 wavLength;
};
struct FFTWop
{
	fftw_complex *in;
	fftw_complex *out;
	fftw_plan p;
	int index;
};
struct FFTW_Result
{
	double* peakfreq;
	double* peakpower;
	double** peakmagMatrix;
	char*** outputMatrix;
	double phase;
};
struct Visual_Pkg
{
  char* Filename;
	int total_packets;
	int total_frames;
	int frame_size;
	int bitsize;

  SDL_AudioDeviceID device;
	SDL_AudioSpec* wavSpec_ptr;
  AudioData* AudioData_ptr;
	FFTW_Result* FFTW_Result_ptr;
	FFTWop* fftw_ptr;


	double (*GetAudioSample)(Uint8*, SDL_AudioFormat);
	void (*setupDFT)(Visual_Pkg*, Uint8*, int );
};

void MyAudioCallback(void* userdata, Uint8* stream, int streamLength);
int runAudio(char* fileName, GPU_Palette* P, CPUAnimBitmap* A);
Visual_Pkg* openAudio(char* fileName);
void printOutput(Visual_Pkg* package);
void myDraw( GPU_Palette* P, CPUAnimBitmap* A, Visual_Pkg* package, int packetN);
int getFileSize(FILE *inFile);
void analyze_FFTW_Results(Visual_Pkg*, struct FFTWop, int, int,size_t);
void getDFT(Visual_Pkg*, Uint8*, int );
void processWAVFile(Uint32 , int , Visual_Pkg* );
float getAmp(Uint8* wavPtr, int start, int end, int offset);
double Get16bitAudioSample(Uint8* bytebuffer, SDL_AudioFormat format);

AudioData* GetAudioData(Visual_Pkg*);
SDL_AudioSpec* GetSDL_AudioSpec(Visual_Pkg*);
FFTW_Result* GetFFTW_Result(Visual_Pkg*);
FFTWop* GetFFTWop(Visual_Pkg*);
void FreePtr(Visual_Pkg* pkg);
#endif // AUDIOliB
