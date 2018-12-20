/*******************************************************************************
*
*   A SOLUTION TO ASSIGNMENT 1, PROBLEM 2, - MICHAEL BRADY
*
*   compile with: gcc -o sol2 solution2.c -lm
*   save results of learn rate = .01 with eg:
*     sol2 .01 > sol2_01.txt
*
*******************************************************************************/
#include <stdio.h>
#include <math.h>
#include "randlib.h"
#include "mnist/mnist.h"

#define gNumInNodes 785       // (28*28)+1 = 785; // extra +1 for bias node
#define gNumOutNodes 10       // training 10 perceptrons
#define gNumEpochs  2         // how many training epochs to train on
#define gNumSimulations 1     // how many simulations to average over
#define gNumLearnRates 2      // test a few differnt learning rates

#define GET_RMS_ERROR 1


// function prototypes:
int checkPrediction(float[], int);
float checkError(float[], float[]);
float mySquash(float);
float testIt(mnist_data*, float[], int);
int trainIt(mnist_data*, float[], float, int);

/******************************************************************************/
int main(int argc, char* argv[])
{
    clock_t start, end;

    // load the MNIST training data
    mnist_data *zTrainData;          // each image is 28x28 pixels
    unsigned int sizeTrainData;      // number if images in data set
    if (mnistLoad(&zTrainData, &sizeTrainData, 0)){ // 60k images
        printf("error loading MNIST training data\n");
        return -1;
    }

    // load the MNIST testing data
    mnist_data *zTestData;          // each image is 28x28 pixels
    unsigned int sizeTestData;      // number if images in data set
    if (mnistLoad(&zTestData, &sizeTestData, 1)){ // 10k images
        printf("error loading MNIST data\n");
        return -1;
    }

    // ---- create, train, and test weights:
    // allocate memory to save results
    float zAnalysisData[gNumEpochs+1][gNumSimulations][gNumLearnRates];
    // allocate memory for weights
    float zWeights[gNumInNodes * gNumOutNodes];

    // run through six different noise levels
	  seed_randoms();

    start = clock();
    for(int p = 0; p < gNumLearnRates; p++){
      float zEta = 1.0 / (pow(10, p));
      for(int sim = 0; sim < gNumSimulations; sim++){

        // randomize weights for the simulation
        for (int i = 0; i < gNumInNodes * gNumOutNodes; i++){
            zWeights[i] = rand_weight();
            }

        // train and test the weights through training epochs
        for(int epoch = 0; epoch < gNumEpochs; epoch++){

            // test weights once before training
            float result;
            if(epoch == 0){ // get results once before any training
              result = testIt(zTestData, zWeights, sizeTestData);
              zAnalysisData[epoch][sim][p] = result;
              }

            // train the weights
            trainIt(zTrainData, zWeights, zEta, sizeTrainData);

            // test the weights after training and save result
            result = testIt(zTestData, zWeights, sizeTestData);
            zAnalysisData[epoch+1][sim][p] = result;
        }
      }
    }
    end = clock();
    printf("time used: %.2f\n", ((double)(end - start))/ CLOCKS_PER_SEC);

    // print results of learning averaged over number of trials
    float lRate;
    for(int p = 0; p < gNumLearnRates; p++){
      lRate = 1.0 / (pow(10, p));
      printf("lRate = %f, ", lRate);
    }
    printf("\n");

    for(int epoch = 0; epoch <= gNumEpochs; epoch++){
      for(int eta = 0; eta < gNumLearnRates; eta++){
        float mean = 0;
        for(int sim = 0; sim < gNumSimulations; sim++){
            mean += zAnalysisData[epoch][sim][eta];
        }
        mean = mean/gNumSimulations;
        printf("%f, ", mean);
      }
      printf("\n");
    }

    free(zTrainData);
    free(zTestData);
    return 0;
}

/******************************************************************************/
int checkPrediction(float pOutput[], int targ){
    // find which idx of the output has the highest value
    float max = 0;
    int idx = 0;
    for(int i = 0; i < gNumOutNodes; i++){
        if(pOutput[i] > max){
            max = pOutput[i];
            idx = i;
        }
    }

    if (idx == targ) return 0;      // correct node had max value
    return 1;                       // incorrect node had max value
}


/******************************************************************************/
float checkError(float pTarget[], float pOutput[])
{
  float error = 0;
  for(int i = 0; i < gNumOutNodes; i++){
    //error += fabs(pTarget[i] - pOutput[i]);
    error += pow(pTarget[i] - pOutput[i], 2);
  }

  return (float) error/gNumOutNodes;
}


/******************************************************************************/
float mySquash(float x)
{
    return 1/(1+(exp(-x)));
}

/******************************************************************************/
float testIt(mnist_data *pData, float pWeights[], int vSizeData)
{
    // allocate memory
    int zIdx;
    int zInputVec[gNumInNodes];
    float zOutput[gNumOutNodes];
    float zTarget[gNumOutNodes];
    float zErrAcc = 0.0;

    // accumulate error for 10k testing images
    for(int cycle = 0; cycle < vSizeData; cycle++){

        // initialize output of all perceptrons to zero
        for (int i = 0; i<gNumOutNodes; i++) zOutput[i] = 0;

        // translate image data to input vector w/ bias node
        get_input(zInputVec, pData, cycle, 0); // 0 means no noise

        // process each perceptron, given the input vector
        for(int j = 0; j < gNumOutNodes; j++){
            for(int i = 0; i < gNumInNodes; i++){
                zOutput[j] += zInputVec[i] * pWeights[(j*gNumInNodes)+i];
            }
            zOutput[j] = mySquash(zOutput[j]);
        }

        if (GET_RMS_ERROR){
          for (int i = 0; i<gNumOutNodes; i++) zTarget[i] = 0;
          zTarget[pData[cycle].label] = 1.0;
          zErrAcc += checkError(zTarget, zOutput);
        }
        else{
          zErrAcc += checkPrediction(zOutput, pData[cycle].label);
        }
    }

    // return average error for the 10k testing images
    return (zErrAcc/(vSizeData * 1.0));
}

/******************************************************************************/
int trainIt(mnist_data *pData, float pWeights[], float vEta, int vSizeData)
{
    // allocate memory
    int zIdx;
    int zInputVec[gNumInNodes];
    float zErr[gNumOutNodes];
    float zOutput[gNumOutNodes];
    float zTarget[gNumOutNodes];

    // do a training epoch for the 60k images
    for(int cycle = 0; cycle < vSizeData; cycle++){

        // initialize output of all perceptrons to zero
        for (int i = 0; i<gNumOutNodes; i++) zOutput[i] = 0;

        // initialize training target (nine nodes off, target node on)
        for (int i = 0; i<gNumOutNodes; i++) zTarget[i] = 0;
        zTarget[pData[cycle].label] = 1;

        // translate image data to input vector w/ bias node
        get_input(zInputVec, pData, cycle, 0); // 0 means no noise

        // process each perceptron, given the input vector
        for(int j = 0; j < gNumOutNodes; j++){
            for(int i = 0; i < gNumInNodes; i++){
                zOutput[j] += zInputVec[i] * pWeights[(j*gNumInNodes)+i];
            }

            zOutput[j] = mySquash(zOutput[j]);
            zErr[j] = zTarget[j] - zOutput[j];

            // update weights based on error
            for(int i = 0; i < gNumInNodes; i++){
                float weightChange = vEta * zErr[j] * zInputVec[i];
                pWeights[(j*gNumInNodes)+i] += weightChange;
            }
        }
    }

    return 0;
}

/*************************************************************************/
