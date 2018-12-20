#include <stdio.h>
#include "randlib.h"
#include "mnist/mnist.h"
#include <math.h>

#define imageSize 785
#define outputSize 2
#define NLayers 100
#define bias 1.0f
int main(int argc, char* argv[])
{
		// --- an example for working with random numbers
		seed_randoms();
		float sampNoise = rand_frac()/2.0; // sets default sampNoise

		// --- a simple example of how to set params from the command line
		if(argc == 2){ // if an argument is provided, it is SampleNoise
				sampNoise = atof(argv[1]);
				if (sampNoise < 0 || sampNoise > .5){
						printf("Error: sample noise should be between 0.0 and 0.5\n");
						return 0;
				}
		}

		// --- an example for how to work with the included mnist library:
		mnist_data *zData;      // each image is 28x28 pixels
		unsigned int sizeData; // depends on loadType
		int loadType = 0; // loadType may be: 0, 1, or 2
		if (mnistLoad(&zData, &sizeData, loadType)){
				printf("something went wrong loading data set\n");
				return -1;
		}

		// loadType = 0, 60k training images
		// loadType = 1, 10k testing images
		// loadType = 2, 10 specific images from training set
		printf("number of training patterns = %d\n", sizeData);

		// inspect the training samples we will work with:
		float inputVec[imageSize]; //(28*28)= 784
		float HiddenLayer1[NLayers];
		float HiddenLayer2[NLayers];
		float WinputToLayers[NLayers * imageSize];
		float WLayer1ToLayer2[NLayers*NLayers];
		float WlayersToOutput[NLayers * outputSize];//there are 785 output nodes

		float output[outputSize];

		float errorOutput[outputSize];
		float errorLayer1[NLayers];
		float errorLayer2[NLayers];

		for(int i = 0; i < NLayers * imageSize; ++i)
		{
			WinputToLayers[i] = rand_weight();
			if(i < NLayers * outputSize) WlayersToOutput[i] = rand_weight();
			if(i < NLayers*NLayers) WLayer1ToLayer2[i] = rand_weight();
		}

		float rate = 0.1f;
		for(int epoch = 0; epoch < 5; ++epoch){
			for(int j = 0; j < sizeData; j++){//based on the size of the data loaded from minst
				get_input(inputVec, zData, j, sampNoise);//input
				//check which number
				//HiddenLayer1
				//---Layer Nodes update from input to Layers--------------------------
				for (int i = 0; i < NLayers; i++){
					HiddenLayer1[i] = 0.0f;// added bias right away
					for(int k = 0; k < imageSize; ++k){//784 pixels
						HiddenLayer1[i] += inputVec[k] * WinputToLayers[k*NLayers + i];//getting output??
					}
					HiddenLayer1[i] = (1.0f/(1.0f + exp(-1.0f * (HiddenLayer1[i]))));//squash
				}
				HiddenLayer1[0] = 1.0f;//bias should be one
				//HiddenLayer2
				//----------------------------------------------------------------------
				for (int i = 0; i < NLayers; i++){
					HiddenLayer2[i] = 0.0f;
					for(int k = 0; k < NLayers; ++k){
						HiddenLayer2[i] += HiddenLayer1[k] * WLayer1ToLayer2[k*NLayers + i];//getting output??
					}
					HiddenLayer2[i] = (1.0f/(1.0f + exp(-1.0f * (HiddenLayer2[i]))));//squash

				}
				HiddenLayer2[0] = 1.0f;//bias should be one

				//---Output Nodes update from input to Layers---------------------------
				for(int i = 0; i < outputSize; ++i)//2
				{
					output[i] = 0.0f;// added bias right away
					for(int k = 0; k < NLayers; ++k){//784 pixels
						output[i] += HiddenLayer2[k] * WlayersToOutput[k*outputSize + i];//getting output??
					}
					output[i] = (1.0f/(1.0f + exp(-1.0f * (output[i]))));//squash
				}
				//----------------------------------------------------------------------

				//BACKPROPAGATION, ERROR--------------------------------------------------------------
				float target1 = zData[j].label % 2;
				float target2 = zData[j].label == 0 ||
				       		zData[j].label == 1 ||
									zData[j].label == 2 ||
									zData[j].label == 3 ||
									zData[j].label == 5 ||
									zData[j].label == 7 ? 1.0f : 0.0f;

				errorOutput[0] = (target1 - output[0]) * output[0] * (1.0f - output[0]);
				errorOutput[1] = (target2 - output[1]) * output[1] * (1.0f - output[1]);

				for(int i = 0; i < NLayers; ++i)
				{
					float sum = 0.0f;
					for(int k = 0; k < outputSize; ++k)
					{
						sum += errorOutput[k]*WlayersToOutput[i*outputSize + k];
					}
					errorLayer2[i] = HiddenLayer2[i]*(1.0f - HiddenLayer2[i])*sum;
				}
				for(int i = 0; i < NLayers; ++i)
				{
					float sum = 0.0f;
					for(int k = 0; k < NLayers; ++k)
					{
						sum += errorLayer2[k]*WLayer1ToLayer2[i*NLayers + k];
					}
					errorLayer1[i] = HiddenLayer1[i]*(1.0f - HiddenLayer1[i])*sum;
				}
				//WeightUpdate-----------------------------------------------------------------------------------
				for (int i = 0; i < NLayers; i++){
					for(int k = 0; k < outputSize; ++k){//784 pixels
						WlayersToOutput[i*outputSize + k] += rate * HiddenLayer2[i] * errorOutput[k];//getting output??
					}
				}

				for (int i = 0; i < NLayers; i++){
					for(int k = 0; k < NLayers; ++k){//784 pixels
						WLayer1ToLayer2[i*NLayers + k] += rate * HiddenLayer1[i] * errorLayer2[k];//getting output??
					}
				}

				for (int i = 0; i < imageSize; i++){
					for(int k = 0; k < NLayers; ++k){//784 pixels
						WinputToLayers[i*NLayers + k] += rate * inputVec[i] * errorLayer1[k];//getting output??
					}
				}
			}
		}

		mnistLoad(&zData, &sizeData, 1);

			for(int j = 0; j < sizeData; j++){//based on the size of the data loaded from minst
				get_input(inputVec, zData, j, sampNoise);//input
				//---Layer Nodes update from input to Layers--------------------------
				for (int i = 0; i < NLayers; i++){
					HiddenLayer1[i] = 0.0f;// added bias right away
					for(int k = 0; k < imageSize; ++k){//784 pixels
						HiddenLayer1[i] += inputVec[k] * WinputToLayers[k*NLayers + i];//getting output??
					}
					HiddenLayer1[i] = (1.0f/(1.0f + exp(-1.0f * (HiddenLayer1[i]))));//squash
				}
				HiddenLayer1[0] = 1.0f;//bias should be one
				//HiddenLayer2
				//----------------------------------------------------------------------
				for (int i = 0; i < NLayers; i++){
					HiddenLayer2[i] = 0.0f;
					for(int k = 0; k < NLayers; ++k){
						HiddenLayer2[i] += HiddenLayer1[k] * WLayer1ToLayer2[k*NLayers + i];//getting output??
					}
					HiddenLayer2[i] = (1.0f/(1.0f + exp(-1.0f * (HiddenLayer2[i]))));//squash

				}
				HiddenLayer2[0] = 1.0f;//bias should be one

				//---Output Nodes update from input to Layers---------------------------
				for(int i = 0; i < outputSize; ++i)//2
				{
					output[i] = 0.0f;// added bias right away
					for(int k = 0; k < NLayers; ++k){//784 pixels
						output[i] += HiddenLayer2[k] * WlayersToOutput[k*outputSize + i];//getting output??
					}
					output[i] = (1.0f/(1.0f + exp(-1.0f * (output[i]))));//squash
				}
				//----------------------------------------------------------------------
				printf("Number = %d\n", zData[j].label);
				printf("Even(0.0f)||Odd(1.0) = %0.5f\n", output[0]);
				printf("Prime number(1.0f): %0.5f\n\n", output[1]);
			}
		free(zData);
		return 0;
}
