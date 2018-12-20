/**************************************************************************
*
*   SOME STARTER CODE FOR WORKING WITH NMIST, © OLENBURG EGOR 2018
*
**************************************************************************/
#include <stdio.h>
#include "randlib.h"
#include "mnist/mnist.h"
#include <math.h>

#define imageSize 785
#define numOutputNodes 785
#define hidden_layer_size  323

void run(int inputVec[], float hiddenL[], float weightVec[], int target[], float output[], float error[], float errorLayer[], mnist_data *zData, float rate, unsigned int sizeData, float sampNoise);
void init(float weightV[], int sz);

void get_output_for_hidenL(float hiddenL[], int inputVec[], float weightInputToLayer[]);
void squashOutput(float vec[], int sz, int beginIndex);
float accumulateCycle_Error(float inputVec[], float output[]);
void get_output(float output[], float hiddenL[], float weightLayerToOutput[]);
void getGlobal_Error(float error[], float inputVec[], float output[], float target);
void getHidden_Error(float error[], float weightLayerToOutput[], float hiddenL[], float errorLayer[]);
void updateWeights_LayerToOutput(float weightLayerToOutput[], float hiddenL[], float error[], float rate);
void updateWeights_InputToLayer(float weightInputToLayer[], float hiddenL[], float error[], float rate);
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
    int weightSize = imageSize * hidden_layer_size + 1; //should be 784 * 10?? since "+1" is bias which is first element of array

    int inputVec[imageSize];
    // float weightVec[weightSize];
    float output[imageSize];
    float error[imageSize]; //tottal error
    float errorLayer[hidden_layer_size];
    float hiddenL[hidden_layer_size +1];//for bias node;
    float weightInputToLayer[weightSize];
    float weightLayerToOutput[weightSize];
    float rate = 0.01f;


    //=====initialize random weights=====//
    init(weightInputToLayer, weightSize);
    init(weightLayerToOutput, weightSize);


    //========running neural network======//
    printf("number of training patterns = %d\n", sizeData);
    printf("learning rate = %f\n", rate);

    for(int epoch = 0; epoch < 20; ++epoch){
          float errAcc = 0.0f;
          float cycleErr;
          float errAcc2 = 0.0;
          float cycleErr2;

          for(int cycle = 0; cycle < sizeData; cycle++){
             cycleErr = 0.0;

             get_input(inputVec, zData, cycle, sampNoise);
             //draw_input( inputVec, zData[cycle].label);

             get_output_for_hidenL(hiddenL, inputVec, weightInputToLayer);
             squashOutput(hiddenL, numOutputNodes, 1);
             get_output(output, hiddenL, weightLayerToOutput);
             squashOutput(output, imageSize, 0);


             errAcc += accumulateCycle_Error(inputVec, output);


             float target = zData[cycle].label % 2;

             cycleErr2 = fabs(target - round(output[0]));
             errAcc2 += cycleErr2;

             getGlobal_Error(error, inputVec, output, target);

             getHidden_Error(error, weightLayerToOutput, hiddenL, errorLayer);


             updateWeights_LayerToOutput(weightLayerToOutput, hiddenL, error, rate);
             updateWeights_InputToLayer(weightInputToLayer, hiddenL,  error,  rate);

          }


        mnistLoad(&zData, &sizeData, 1);

        for(int cycle = 0; cycle < sizeData; cycle++){
           cycleErr = 0.0;

           get_input(inputVec, zData, cycle, sampNoise);

           get_output_for_hidenL(hiddenL, inputVec, weightInputToLayer);
           squashOutput(hiddenL, numOutputNodes, 1);
           get_output(output, hiddenL, weightLayerToOutput);
           squashOutput(output, imageSize, 0);


           errAcc += accumulateCycle_Error(inputVec, output);

           float target = zData[cycle].label % 2;

           cycleErr2 = fabs(target - round(output[0]));
           errAcc2 += cycleErr2;

         }
         printf("%f\n", errAcc/sizeData);



    }

    printf("\n");
    free(zData);
    return 0;
}


//======init rand weights==========//

void init(float weightV[], int sz)
{
    for(int i = 0; i < sz; ++i)
    {
       weightV[i] = rand_weight();
    }
}
//======end init weights =========//

//=======getting output========//
// calculate output by multiplying matrix of inputs and weights
void get_output_for_hidenL(float hiddenL[], int inputVec[], float weightInputToLayer[])
{
    for (int i = 0; i < hidden_layer_size; i++){ //10
        hiddenL[i] = 0.0f;
        for(int j = 0; j < imageSize; ++j){ //785
          hiddenL[i] += inputVec[j] * weightInputToLayer[i*imageSize + j];
        }
    }
    hiddenL[0] = 1.0f;//bias node;

}
//======end getting output=====//

//======squashing output======//
void squashOutput(float vec[], int sz, int beginIndex)
{
  for(int index = beginIndex; index < sz; index++)
  {
      vec[index] = (1.0/(1.0 + exp(-1.0f * vec[index])));
  }

}
//======end squash output ====//

//=======accumulate error for cycle=====//
float accumulateCycle_Error(float inputVec[], float output[])
{
    float sum  = 0.0;
    for(int k = 0; k < imageSize; k++){
        sum += fabs(inputVec[k] - output[k])/imageSize;
    }
    return sum;
}
//======end of error acc for cycle======//

//============get total output=======//
void get_output(float output[], float hiddenL[], float weightLayerToOutput[])
{
    for (int i = 0; i < imageSize; i++){ //785
        output[i] = 0.0f;
        for(int j = 0; j < hidden_layer_size + 1; ++j){ //11
            output[i] += hiddenL[j] * weightLayerToOutput[i*hidden_layer_size + j]; //1 * 785 + 1 = weightVec[786]...;
        }
    }
    // output[0] = 1.0f; //bias node;

}
//==========end get total output=====//

//=========get global error=========//

void getGlobal_Error(float error[], float inputVec[], float output[], float target)
{
    error[0] = (target - output[0]) * output[0] * (1 - output[0]);
    //printf("oddEven = %d, imgNum = %d, output[0] = %f, inputVec[0] = %f\n", zTarget, target, output[0], inputVec[0]);

    for(int e = 1; e < imageSize; e++)
    {
        error[e] = (inputVec[e] - output[e])*output[e]*(1 - output[0]);
    }
}
//=========end get global error=====//

//======getting error=========//
void getHidden_Error(float error[], float weightLayerToOutput[], float hiddenL[], float errorLayer[])
{
    for(int j = 0; j < hidden_layer_size; j++) errorLayer[j] = 0.0; // zero array

    for(int k = 0; k < imageSize; k++)
    {
        for(int j = 1; j < hidden_layer_size + 1; j++) // accumulate hidden error,
        {
           // ignoring bias node that is in weightLayerToOutput at index 0
              errorLayer[j-1] += error[k] * weightLayerToOutput[k * hidden_layer_size + j];
        }
    }
    //unsquashing hidden error;
    for(int j = 0; j < hidden_layer_size; j++){
      errorLayer[j] *= (hiddenL[j+1] * (1.0 - hiddenL[j+1]));
    }

}
//======end getting error=====//

//======= updating weights from layer to output ====//
void updateWeights_LayerToOutput(float weightLayerToOutput[], float hiddenL[], float error[], float rate)
{
    for(int w = 0; w < imageSize; ++w){
       for(int w2 = 0; w2 < hidden_layer_size;  w2++)
       {
         weightLayerToOutput[w*hidden_layer_size + w2] += rate * hiddenL[w2] * error[w];
       }
    }

}
//=======end updating weights from layer to output====//

//======= updating weights from input to layer====//

void updateWeights_InputToLayer(float weightInputToLayer[], float hiddenL[], float error[],  float rate)
{
    for(int w = 0; w < imageSize; ++w){
      for(int w2 = 0; w2 < hidden_layer_size;  w2++)
      {
        weightInputToLayer[w*hidden_layer_size + w2] += rate * hiddenL[w2] * error[w];
      }
    }

}
//=======end updating weights from input to layer====//

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
