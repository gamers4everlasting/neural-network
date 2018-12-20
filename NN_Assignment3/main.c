/**************************************************************************
*
*   SOME STARTER CODE FOR WORKING WITH NMIST, © OLENBURG EGOR 2018
*
* use command to run:  gcc main.c -lm -o main
**************************************************************************/
#include <stdio.h>
#include "randlib.h"
#include "mnist/mnist.h"
#include <math.h>

#define numInputNodes 785
#define numOutputNodes 2
#define hidden_layer_size1  50
#define hidden_layer_size2 50

void init(float weightV[], int sz);
void get_output(float layer[], float vec[], float weights[], int sz1, int sz2);
void getGlobal_Error(float error[], float output[], int target);
void getHidden_Error(float error[], float weights[], float hiddenL[], float errorLayer[], int szL, int szO);
void updateWeights(float weights[], float layer[], float error[], float rate, int sz1, int sz2);
float getPrime(int t);
float accumulateCycle_Error(float inputVec[], float output[]);


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
    int loadType = 1; // loadType may be: 0, 1, or 2
    if (mnistLoad(&zData, &sizeData, loadType)){
        printf("something went wrong loading data set\n");
        return -1;
    }



    //====data size=====//
    int weightInputToLayerSize = numInputNodes * hidden_layer_size1;
    int weightLayerToLayerSize = hidden_layer_size1 * hidden_layer_size2;
    int weightLayerToOutputSize = hidden_layer_size2 * numOutputNodes;

    //===== data ====//
    float inputVec[numInputNodes];
    float output[numOutputNodes]; //final output
    float error[numOutputNodes]; //final error
    float errorLayer1[hidden_layer_size1];
    float errorLayer2[hidden_layer_size2];
    float hiddenLayerOutput1[hidden_layer_size1];//[0] is bias node;
    float hiddenLayerOutput2[hidden_layer_size2];//[0] is bias node;
    float weightInputToLayer1[weightInputToLayerSize];
    float weightLayer1ToLayer2[weightLayerToLayerSize];
    float weightLayer2ToOutput[weightLayerToOutputSize];
    float target[numOutputNodes];
    float rate = 0.5f;


    //=====initialize random weights=====//
    init(weightInputToLayer1, weightInputToLayerSize);
    init(weightLayer1ToLayer2, weightLayerToLayerSize);
    init(weightLayer2ToOutput, weightLayerToOutputSize);


    //========running neural network======//
    printf("number of training patterns = %d\n", sizeData);
    printf("learning rate = %f\n", rate);


   for(int epoch = 0; epoch < 5; ++epoch){
         float errAcc = 0.0f;

         for(int cycle = 0; cycle < sizeData; cycle++){

            get_input(inputVec, zData, cycle, sampNoise);

            get_output(hiddenLayerOutput1, inputVec, weightInputToLayer1, hidden_layer_size1, numInputNodes);
            hiddenLayerOutput1[0] = 1.0f;
            get_output(hiddenLayerOutput2, hiddenLayerOutput1, weightLayer1ToLayer2, hidden_layer_size2, hidden_layer_size1);
            hiddenLayerOutput2[0] = 1.0f;
            get_output(output, hiddenLayerOutput2, weightLayer2ToOutput, numOutputNodes,  hidden_layer_size2);

            target[0] = zData[cycle].label % 2 == 0 ? 1:0;
            target[1] = getPrime(zData[cycle].label);


            errAcc += accumulateCycle_Error(target,  output);

            // printf("Number = %d\n", zData[cycle].label);
            // printf("Even(1.0f)||Odd(0.0f) = %0.5f\n", output[0]);
            // printf("Prime number(1.0f): %0.5f\n", output[1]);

          }
          printf("%f\n", errAcc / sizeData);

          mnistLoad(&zData, &sizeData, 0);
          for(int cycle = 0; cycle < sizeData; cycle++){

             get_input(inputVec, zData, cycle, sampNoise);

             get_output(hiddenLayerOutput1, inputVec, weightInputToLayer1, hidden_layer_size1, numInputNodes);
             hiddenLayerOutput1[0] = 1.0f; //bias nodes
             get_output(hiddenLayerOutput2, hiddenLayerOutput1, weightLayer1ToLayer2, hidden_layer_size2, hidden_layer_size1);
             hiddenLayerOutput2[0] = 1.0f; //bias node
             get_output(output, hiddenLayerOutput2, weightLayer2ToOutput, numOutputNodes,  hidden_layer_size2);





             getGlobal_Error(error, output, zData[cycle].label);
             getHidden_Error(error, weightLayer2ToOutput, hiddenLayerOutput2, errorLayer2, hidden_layer_size2, numOutputNodes);
             getHidden_Error(errorLayer2, weightLayer1ToLayer2, hiddenLayerOutput1, errorLayer1, hidden_layer_size1, hidden_layer_size2);


             updateWeights(weightLayer2ToOutput, hiddenLayerOutput2, error, rate, hidden_layer_size2, numOutputNodes);
             updateWeights(weightLayer1ToLayer2, hiddenLayerOutput1,  errorLayer2,  rate, hidden_layer_size1, hidden_layer_size2);
             updateWeights(weightInputToLayer1, inputVec,  errorLayer1,  rate,  numInputNodes, hidden_layer_size1);
          }




      }
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
void get_output(float layer[], float vec[], float weights[], int sz1, int sz2)
{

    for (int i = 0; i < sz1; i++){ //hid layer size
        layer[i] = 0.0f;
        for(int j = 0; j < sz2; ++j){ //numInputNodes
          layer[i] += vec[j] * weights[j*sz1 + i];
        }
        layer[i] = (1.0f/(1.0f + exp(-1.0f * layer[i])));
    }
}
//======end getting output=====//


//=========get global error=========//

void getGlobal_Error(float error[], float output[], int target)
{
    float t = target % 2==0 ? 1: 0;
    error[0] = (t - output[0]) * output[0] * (1.0f - output[0]);

    t = getPrime(target);
    error[1] = (t - output[1]) * output[1] * (1.0f - output[1]);
}
//=========end get global error=====//

//======getting hidden error=========//
void getHidden_Error(float error[], float weights[], float hiddenL[], float errorLayer[], int szL, int szO)
{
    for(int j = 0; j < szL; j++) errorLayer[j] = 0.0; // zero array

    for(int k = 0; k < szL; k++)//50  50
    {
        for(int j = 0; j < szO; j++)//2,  50
        {
            errorLayer[k] += error[j] * weights[k * szO + j];
        }
        errorLayer[k] *= (hiddenL[k] * (1.0 - hiddenL[k]));  //unsquashing hidden error;
    }
}
//======end getting  hidden  error=====//

//======= updating weights all====//
void updateWeights(float weights[], float layer[], float error[], float rate, int sz1, int sz2)
{
    for(int w = 0; w < sz1; ++w){ //hid nodes
       for(int w2 = 0; w2 < sz2;  w2++) //num inputVec
       {
         weights[w*sz2 + w2] += rate * layer[w] * error[w2];
       }
    }

}
//=======end updating weights from layer to output====//

//========getting odd or even target=====//
float getPrime(int t)
{
  if (t == 0 || t == 1 || t == 2 || t== 3 || t== 5 || t == 7)
  {
    return 1.0f;
  }
  return 0.0f;
}

//=======accumulate error for cycle=====//
float accumulateCycle_Error(float target[], float output[])
{
    float sum  = 0.0;
    for(int k = 0; k < numOutputNodes; k++){
        sum += powf((target[k] - output[k]), 2);
    }
    return (sum / numOutputNodes);
}
//======end of error acc for cycle======//
