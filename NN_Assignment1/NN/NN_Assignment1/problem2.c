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
#define numOutputNodes 10

void run(int inputVec[], float weightVec[], int target[], float output[], float error[], mnist_data *zData, float rate, unsigned int sizeData, float sampNoise);
void get_output(float output[], int inputVec[], float weightVec[]);
void squashOutput(float output[]);
void getError(float error[], float output[], int picIndex);
void updateWeights(int inputVec[], float error[], float weightVec[], float rate);
float getAverageError(float error[]);

int main(int argc, char* argv[])
{
    // --- an example for working with random numbers
    seed_randoms();
    float sampNoise = 0;
    //--- a simple example of how to set params from the command line
    if(argc == 2){ // if an argument is provided, it is SampleNoise
        sampNoise = atof(argv[1]);
        if (sampNoise < 0 || sampNoise > .5){
            printf("Error: sample noise should be between 0.0 and 0.5\n");
            return 0;
        }
    }

    // --- an example for how to work with the included mnist library:
    // loadType = 0, 60k training images
    // loadType = 1, 10k testing images
    // loadType = 2, 10 specific images from training set
    mnist_data *zData;      // each image is 28x28 pixels
    unsigned int sizeData; // depends on loadType
    int loadType = 0; // loadType may be: 0, 1, or 2
    if (mnistLoad(&zData, &sizeData, loadType)){
        printf("something went wrong loading data set\n");
        return -1;
    }



    //====data=====//
    int weightSize = imageSize * numOutputNodes; //should be 784 * 10?? since "+1" is bias which is first element of array
    int inputVec[imageSize];
    float weightVec[weightSize];
    int target[numOutputNodes];
    float output[numOutputNodes];
    float error[numOutputNodes];
    float rate = 0.5f;


    //=====initialize random weights=====//
    for(int i = 0; i < weightSize; ++i)
    {
	     weightVec[i] = rand_weight();
    }

    //========running neural network======//
    printf("number of training patterns = %d\n", sizeData);
    printf("learning rate = %f\n", rate);

    run(inputVec, weightVec, target, output, error, zData, rate, sizeData, sampNoise);

    free(zData);
    return 0;
}

//======run neural network =====//
void run(int inputVec[], float weightVec[], int target[], float output[], float error[], mnist_data *zData, float rate, unsigned int sizeData, float sampNoise)
{
    for(int epoch = 0; epoch < 20; ++epoch){
          float epochError = 0;
          // printf("Learning epoch: %d\n", epoch);

          for(int pic = 0; pic < sizeData; pic++){

             get_input(inputVec, zData, pic, sampNoise);
             get_output(output, inputVec, weightVec);
             squashOutput(output);
             getError(error, output, zData[pic].label);
             updateWeights(inputVec, error, weightVec, rate);
             epochError += getAverageError(error);

          }
          printf("%f,\n", (epochError / sizeData));

    }
    printf("\n");
}

//======end run neural network =====//


//=======getting output========//
// calculate output by multiplying matrix of inputs and weights
void get_output(float output[], int inputVec[], float weightVec[])
{
    for (int i = 0; i < numOutputNodes; i++){ //10
        output[i] = 0.0f;
        for(int j = 0; j < imageSize; ++j){ //785
          output[i] += inputVec[j] * weightVec[j*numOutputNodes + i]; //1 * 785 + 1 = weightVec[786]...;
        }
    }
}
//======end getting output=====//

//======squashing output======//
void squashOutput(float output[])
{
  for(int index = 0; index < numOutputNodes; index++)
  {
      output[index] = (1.0/(1.0 + exp(-1.0f * output[index])));
  }

}
//======end squash output ====//

//======getting error=========//
void getError(float error[], float output[], int picIndex)
{
  for(int i = 0; i < numOutputNodes; i++)
  {
    error[i] = i == picIndex ? 1.0f - output[i] : 0.0f - output[i];

  }
}

//======end getting error=====//
void updateWeights(int inputVec[], float error[], float weightVec[], float rate)
{
    for(int i = 0; i < numOutputNodes; i++)
    {
        for(int j = 0; j < imageSize; ++j)
        {
            weightVec[j*numOutputNodes + i] += rate * inputVec[j] * error[i];
        }
    }
}
//=======end getting error====//

//=========getting average error======//
float getAverageError(float error[])
{
    float sum = 0.0f;
    for(int i = 0; i < numOutputNodes; i++)
    {
        sum += fabs(error[i]);
    }
    return (sum / numOutputNodes);
}
//========end getting average error======//
