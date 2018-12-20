/**************************************************************************
*
*   SOME STARTER CODE FOR WORKING WITH NMIST, © MICHAEL BRADY 2018
*
**************************************************************************/
#include <stdio.h>
#include "randlib.h"
#include "mnist/mnist.h"
#include <math.h>

#define imageSize 785
#define N 10
#define hidden_layer_size  10


int main(int argc, char* argv[])
{
    // --- an example for working with random numbers
    seed_randoms();
    float sampNoise = 0;//rand_frac()/2.0; // sets default sampNoise
    float rate = 0.5f;
    //float answer [10][20];
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
    int loadType = 2; // loadType may be: 0, 1, or 2
    if (mnistLoad(&zData, &sizeData, loadType)){
        printf("something went wrong loading data set\n");
        return -1;
    }

    // loadType = 0, 60k training images
    // loadType = 1, 10k testing images
    // loadType = 2, 10 specific images from training set
    printf("number of training patterns = %d\n", sizeData);
    printf("learning rate = %f\n", rate);

    // inspect the training samples we will work with:
    int inputVec[imageSize]; //(28*28)+1 = 785
    float hiddenL[hidden_layer_size];
    int weightSizeLayer = imageSize * hidden_layer_size; //should be 784 * 10?? since "+1" is bias which is first element of array
    int weightSize = imageSize * N;
    float weightInputToLayer[weightSizeLayer];
    float weightLayerToOutput[weightSize];// +1: for extra od or evern node;
    for(int i = 0; i < weightSize; ++i)
    {
      weightLayerToOutput[i] = rand_weight();
      weightInputToLayer[i] = rand_weight();
    }

    float output[weightSize];

  for(int epoch = 0; epoch < 20; ++epoch){
      float ErrorLayer[N];
      float ErrorOutput[imageSize];
      printf("Learning cycle: %d\n", epoch);
      printf("this is an extra line\n");
      for(int j = 0; j < 10; j++){ //train data each
  	     get_input(inputVec, zData, j, sampNoise);
  	//draw_input(inputVec, zData[j].label);
  	//printf("Number = %d\n", zData[j].label);
      	for (int i = 0; i < N; i++){
        		hiddenL[i] = 0.0f;
        		for(int k = 1; k < imageSize; ++k){ //no need to add bias
        			   hiddenL[i] += inputVec[k] * weightInputToLayer[k*N + i];//getting output
        		}

            //squashing hiddenlayer
        		hiddenL[i] = (1.0/(1.0 + exp(-1.0f * hiddenL[i])));
            printf("squashed hiddenL = %f\n", hiddenL[i]);
            // get our output
            for (int i = 0; i < imageSize; i++){ //+1: for od, even NODE;
                output[i] = 1.0f;
                for(int k = 0; k < N; ++k){
                     output[i] += hiddenL[k] * weightLayerToOutput[k*(imageSize) + i];//getting output
                }
            }
            //squash
            //getting target: if label % 2 == 0 then 0, else 1;
            float target = zData[j].label % 2;
            printf("target = %d\n",  zData[j].label);
            //get error for target if even or odd
            ErrorOutput[0] = (target - output[0]) * output[0] * (1 - output[0]);

            //geting global error
            for(int k = 1; k < imageSize; k++)
            {
                ErrorOutput[k] = (inputVec[k] - output[k])*output[k]*(1 - output[0]);
            }
            //getting layer Error
            for(int e = 0; e < N; e++)
            {
              float errorSum = 0;
              for(int e2 = 0; e2 < imageSize; e2 ++)
              {
                 errorSum += ErrorOutput[e2] * weightLayerToOutput[e * imageSize + e2];//0-785
              }

              ErrorLayer[e] = hiddenL[e] * (1 - hiddenL[e]) * errorSum;
            }

            //updating weights output to layer
           for(int w = 0; w < hidden_layer_size; ++w){
              for(int w2 = 0; w2 < imageSize;  w2++)
              {
                weightLayerToOutput[w*imageSize + w2] += rate * hiddenL[w] * ErrorOutput[w2];
              }
           }
            //updating weights from layer to input
            for(int w = 0; w < imageSize; ++w){
              for(int w2 = 0; w2 < hidden_layer_size;  w2++)
              {
                weightInputToLayer[w*hidden_layer_size + w2] += rate * hiddenL[w2] * ErrorOutput[w];
              }
            }


        		//printf("output[%d] = %f, error[%d] = %f\n", i, output[i], zData[j].label, Error[zData[j].label]);
        }
        	//geting average error
      	// float sum = 0.0f;
      	// for(int i = 0; i < N; i++)
      	// {
        //   //printf("error[%d] = %f\n", i, fabs(Error[i]));
      	// 	sum += fabs(Error[i]);
      	// }
      	// printf("%f,\n", sum / N);
        // //answer[epoch][j] = sum/N;
        }



    }
    free(zData);
    return 0;
}
