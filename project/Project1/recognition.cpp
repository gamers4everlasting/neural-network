#include <stdio.h>
#include <cstdlib>		// need this?
#include <string.h>		// need this?
#include <time.h>
#include <unistd.h> 	// includes usleep
#include <SDL2/SDL.h>	// for sound processing..
#include <math.h>
#include <fstream>
#include <istream>
#include <sys/types.h>
#include <dirent.h>
#include <vector>
#include <string>

#include "randlib.h"
#include "audio.h"
#include "crack.h"

volatile int packet_pos = 0;
volatile int print_spectrum = 0;
static volatile int time_to_exit = 0;


#define inputLayerSize 162 //min packetsize is 162
#define outputLayerSize 10
#define hiddenLayerSize 70
#define bias 1.0f
/*
10 genres:
blues 0
classical 1
country 2
disco 3
hiphop 4
jazz 5
metal 6
pop 7
reggae 8
rock 9
*/
float learningRate = 0.1f;
float HiddenLayer1[hiddenLayerSize];
float HiddenLayer2[hiddenLayerSize];
float WInputToLayer1[hiddenLayerSize * inputLayerSize];
float WLayer1ToLayer2[hiddenLayerSize* hiddenLayerSize];
float WLayer2ToOutput[hiddenLayerSize * outputLayerSize];//there are 785 output nodes
float output[outputLayerSize];
float errorOutput[outputLayerSize];
float errorLayer1[hiddenLayerSize];
float errorLayer2[hiddenLayerSize];
//float targets[outputLayerSize] = {440.00f, 466.16f, 493.88f, 523.25f, 554.37f, 587.33f, 622.25f, 659.25f, 698.46f, 739.99f, 783.99f, 830.61f};
//float ratio = 1.059463094359;
int TARGET = 0;
void openDirectory(std::vector<char*>& fileList, char* path)
{
  DIR *directory = opendir(path);
  dirent *entry;
  if(directory == nullptr)
  {
    return;
  }

  while(entry = readdir(directory))
  {
    if(entry->d_type != DT_DIR)
    {
      fileList.push_back(entry->d_name);
    }
  }
}
void saveWeights()
{
  std::ofstream InputWeightsLayer1("InputWeights.txt");
  if(InputWeightsLayer1.is_open()){
    for(int i = 0; i < hiddenLayerSize * inputLayerSize; ++i)
    {
      InputWeightsLayer1 << WInputToLayer1[i] << std::endl;
    }
    InputWeightsLayer1.close();
  }

  std::ofstream Layer1WeightsLayer2("Layer1Weights.txt");
  if(Layer1WeightsLayer2.is_open()){
    for(int i = 0; i < hiddenLayerSize* hiddenLayerSize; ++i)
    {
      Layer1WeightsLayer2 << WLayer1ToLayer2[i] << std::endl;
    }
    Layer1WeightsLayer2.close();
  }

  std::ofstream Layer2WeightsOutput("Layer2Weights.txt");
  if(Layer2WeightsOutput.is_open()){
    for(int i = 0; i < hiddenLayerSize * outputLayerSize; ++i)
    {
      Layer2WeightsOutput << WLayer2ToOutput[i] << std::endl;
    }
    Layer2WeightsOutput.close();
  }
  return;
}
void rewriteWeights()
{
  std::ofstream InputWeightsLayer1("InputWeights.txt", std::ofstream::trunc);
  if(InputWeightsLayer1.is_open()){
    for(int i = 0; i < hiddenLayerSize * inputLayerSize; ++i)
    {
      InputWeightsLayer1 << WInputToLayer1[i] << std::endl;
    }
    InputWeightsLayer1.close();
  }

  std::ofstream Layer1WeightsLayer2("Layer1Weights.txt", std::ofstream::trunc);
  if(Layer1WeightsLayer2.is_open()){
    for(int i = 0; i < hiddenLayerSize* hiddenLayerSize; ++i)
    {
      Layer1WeightsLayer2 << WLayer1ToLayer2[i] << std::endl;
    }
    Layer1WeightsLayer2.close();
  }

  std::ofstream Layer2WeightsOutput("Layer2Weights.txt", std::ofstream::trunc);
  if(Layer2WeightsOutput.is_open()){
    for(int i = 0; i < hiddenLayerSize * outputLayerSize; ++i)
    {
      Layer2WeightsOutput << WLayer2ToOutput[i] << std::endl;
    }
    Layer2WeightsOutput.close();
  }
  return;
}
void openWeights()
{

  std::ifstream InputWeightsLayer1;
  InputWeightsLayer1.open("InputWeights.txt");
  if(InputWeightsLayer1.is_open()){
    float readN;
    for(int i = 0; i < hiddenLayerSize * inputLayerSize; ++i)
    {
      InputWeightsLayer1 >> readN;
      WInputToLayer1[i] = readN;
    }
    InputWeightsLayer1.close();
  }

  std::ifstream Layer1WeightsLayer2;
  Layer1WeightsLayer2.open("Layer1Weights.txt");
  if(Layer1WeightsLayer2.is_open()){
    float readN;
    for(int i = 0; i < hiddenLayerSize* hiddenLayerSize; ++i)
    {
      Layer1WeightsLayer2 >> readN;
      WLayer1ToLayer2[i] = readN;
    }
    Layer1WeightsLayer2.close();
  }

  std::ifstream Layer2WeightsOutput;
  Layer2WeightsOutput.open("Layer2Weights.txt");
  if(Layer2WeightsOutput.is_open()){
    float readN;
    for(int i = 0; i < hiddenLayerSize * outputLayerSize; ++i)
    {
      Layer2WeightsOutput >> readN;
      WLayer2ToOutput[i] = readN;
    }
    Layer2WeightsOutput.close();
  }
  return;
}
void initRandomWeights()
{
	for(int i = 0; i < hiddenLayerSize * inputLayerSize; ++i)
	{
		WInputToLayer1[i] = rand_weight();
		if(i < hiddenLayerSize * outputLayerSize) WLayer2ToOutput[i] = rand_weight();
		if(i < hiddenLayerSize*hiddenLayerSize) WLayer1ToLayer2[i] = rand_weight();
	}
}
void train(FFTW_Result* frequencies, int target)
{

	//HiddenLayer1
	//---Layer Nodes update from input to Layers--------------------------
	for (int i = 0; i < hiddenLayerSize; i++){
		HiddenLayer1[i] = 0.0f;// added bias right away
		for(int k = 0; k < inputLayerSize; ++k)
    {//784 pixels
			HiddenLayer1[i] += frequencies[k].peakfreq[0] * WInputToLayer1[k*hiddenLayerSize + i];
      //getting output??
		}
		HiddenLayer1[i] = (1.0f/(1.0f + exp(-1.0f * (HiddenLayer1[i]))));
	}
	HiddenLayer1[0] = 1.0f;//bias should be one
	//HiddenLayer2
	//----------------------------------------------------------------------
	for (int i = 0; i < hiddenLayerSize; i++){
		HiddenLayer2[i] = 0.0f;
		for(int k = 0; k < hiddenLayerSize; ++k)
    {
			HiddenLayer2[i] += HiddenLayer1[k] * WLayer1ToLayer2[k*hiddenLayerSize + i];//getting output??
    }
    HiddenLayer2[i] = (1.0f/(1.0f + exp(-1.0f * (HiddenLayer2[i]))));//squash
  }
	HiddenLayer2[0] = 1.0f;//bias should be one

	//---Output Nodes update from input to Layers---------------------------
	for(int i = 0; i < outputLayerSize; ++i)//2
	{
		output[i] = 0.0f;// added bias right away
		for(int k = 0; k < hiddenLayerSize; ++k)
    {//784 pixels
			output[i] += HiddenLayer2[k] * WLayer2ToOutput[k*outputLayerSize + i];//getting output??
	  }
		output[i] = (1.0f/(1.0f + exp(-1.0f * (output[i]))));//squash
  }
	//----------------------------------------------------------------------

	//BACKPROPAGATION, ERROR--------------------------------------------------------------
	for(int i = 0; i < outputLayerSize; ++i)
	{
    float check = target == i ? 1.0f : 0.0f;
		errorOutput[i] = (check - output[i]) * output[i] * (1.0f - output[i]);
	}

	for(int i = 0; i < hiddenLayerSize; ++i)
	{
		float sum = 0.0f;
		for(int k = 0; k < outputLayerSize; ++k)
		{
			sum += errorOutput[k]*WLayer2ToOutput[i*outputLayerSize + k];
		}
		errorLayer2[i] = HiddenLayer2[i]*(1.0f - HiddenLayer2[i])*sum;
	}
	for(int i = 0; i < hiddenLayerSize; ++i)
	{
		float sum = 0.0f;
		for(int k = 0; k < hiddenLayerSize; ++k)
		{
			sum += errorLayer2[k]*WLayer1ToLayer2[i*hiddenLayerSize + k];
		}
		errorLayer1[i] = HiddenLayer1[i]*(1.0f - HiddenLayer1[i])*sum;

	}
	//WeightUpdate-----------------------------------------------------------------------------------
	for (int i = 0; i < hiddenLayerSize; i++){
		for(int k = 0; k < outputLayerSize; ++k)
    {//784 pixels
			WLayer2ToOutput[i*outputLayerSize + k] += learningRate * HiddenLayer2[i] * errorOutput[k];//getting output??
		}
	}

	for (int i = 0; i < hiddenLayerSize; i++){
		for(int k = 0; k < hiddenLayerSize; ++k)
    {//784 pixels
			WLayer1ToLayer2[i*hiddenLayerSize + k] += learningRate * HiddenLayer1[i] * errorLayer2[k];//getting output??
		}
	}

	for (int i = 0; i < inputLayerSize; i++){
		for(int k = 0; k < hiddenLayerSize; ++k)
    {//784 pixels
			WInputToLayer1[i*hiddenLayerSize + k] += learningRate * frequencies[i].peakfreq[0] * errorLayer1[k];//getting output??
		}
	}

	return;
}

void check(FFTW_Result* frequencies)
{
	for (int i = 0; i < hiddenLayerSize; i++){
		HiddenLayer1[i] = 0.0f;// added bias right away
		for(int k = 0; k < inputLayerSize; ++k){//784 pixels
			HiddenLayer1[i] += frequencies[k].peakfreq[0] * WInputToLayer1[k*hiddenLayerSize + i];//getting output??
		}
		HiddenLayer1[i] = (1.0f/(1.0f + exp(-1.0f * (HiddenLayer1[i]))));//squash
	}
	HiddenLayer1[0] = 1.0f;//bias should be one
	//HiddenLayer2
	//----------------------------------------------------------------------
	for (int i = 0; i < hiddenLayerSize; i++){
		HiddenLayer2[i] = 0.0f;
		for(int k = 0; k < hiddenLayerSize; ++k)
    {
			HiddenLayer2[i] += HiddenLayer1[k] * WLayer1ToLayer2[k*hiddenLayerSize + i];//getting output??
    }
		HiddenLayer2[i] = (1.0f/(1.0f + exp(-1.0f * (HiddenLayer2[i]))));//squash
	}
	HiddenLayer2[0] = 1.0f;//bias should be one

	//---Output Nodes update from input to Layers---------------------------
	for(int i = 0; i < outputLayerSize; ++i)//2
	{
		output[i] = 0.0f;// added bias right away
		for(int k = 0; k < hiddenLayerSize; ++k){//784 pixels
			output[i] += HiddenLayer2[k] * WLayer2ToOutput[k*outputLayerSize + i];//getting output??
		}
		output[i] = (1.0f/(1.0f + exp(-1.0f * (output[i]))));//squash
	}

  for(int i = 0; i < outputLayerSize; ++i)
  {
    printf("%.2f\n", output[i]);
  }
	return;
}

int main(int argc, char *argv[])
{
  openWeights();
  unsigned char ch; // must be at least one arg (fileName)

  if(argc<2){return 1;} // must be at least one arg (fileName)
	while((ch = crack(argc, argv, "t|f|", 0)) != NULL) {
	  switch(ch){
    	case 't' : TARGET = atoi(arg_option); break;
      default  : return(0);
    }
  }

  printf("%d\n", TARGET);
  char* path = argv[arg_index];
  std::vector<char*> list;
  openDirectory(list, path);
  for(int i = 0; i < list.size(); ++i){
    char fileStr[255];
    strcpy(fileStr, path);
    strcat(fileStr, list[i]);
    printf("%s\n", fileStr);
    Visual_Pkg* pkg = openAudio(fileStr);
    FFTW_Result* res= GetFFTW_Result(pkg);

    check(pkg->FFTW_Result_ptr);

     for(int epoch = 0; epoch < 10; ++epoch){
       train(pkg->FFTW_Result_ptr, TARGET);
     }
    printf("Filename: %s\n", pkg->Filename);
    printf("Total packets: %d\n", pkg->total_packets);
    printf("Total frames: %d\n", pkg->total_frames);
    printf("Frame size: %d\n", pkg->frame_size);
    printf("Bitsize: %d\n", pkg->bitsize);
    struct FFTWop* fftwop = GetFFTWop(pkg);

  }
  rewriteWeights();

  return 0;
}


for (int i = 0; i <  pkg->total_packets; ++i){
    //for peak results
      printf("i = %d with peak result of %f and ", i, pkg->FFTW_Result_ptr[i].peakfreq[0]);
      printf(" peak power of %f\n",pkg->FFTW_Result_ptr[i].peakpower[0]);
    printf("----------------------------\n");
    for(int j = 0; j< 5; ++j){
        printf("Peak max result: %f\n", pkg->FFTW_Result_ptr[i].peakmagMatrix[ch][j]);// to fit the result into console screen;                                                              // use pow(peakmagMatrix[ch][i],10)

    }
    printf("**************************\n");
}

