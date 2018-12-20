/*******************************************************************************
*
*   A SOLUTION TO ASSIGNMENT 1, PROBLEM 1, - MICHAEL BRADY
*
*   compile with: gcc -o sol1 solution1.c -lm
*   save results of learn rate = .01 with eg:
*     sol1 .01 > results_01.txt
*
*******************************************************************************/
#include <stdio.h>
#include <math.h>
#include "randlib.h"
#include "mnist/mnist.h"

#define gNumInNodes 785       // (28*28)+1 = 785; // extra +1 for bias node
#define gNumOutNodes 10       // training 10 perceptrons
#define gNumEpochs  100       // how many training epochs per perceptron
#define gNumSimulations 20    // how many trials to average results over
#define gNumNoiseLevels 6     // {0, .1, .2, .3, .4, .5}

// the ten specific example digits we will use for Assignment 1, Problem 1
//static int gSamples[10] = {56, 6, 213, 7, 58, 173, 90, 214, 245, 226};

// function prototypes:
int checkPrediction(float [], int);
float checkErr(float []);
float mySquash(float);
float runIt(mnist_data *, float [], float, float, int);

/******************************************************************************/
int main(int argc, char* argv[])
{
    // read learn rate (eta) as command line argument
    float zEta = .1; // set default learn rate
    if(argc == 2){ // if an argument is provided, use that as eta instead
        zEta = atof(argv[1]);
    }

    // load the MNIST data
    mnist_data *zData;          // each image is 28x28 pixels
    unsigned int sizeData;      // number if images in data set
    if (mnistLoad(&zData, &sizeData, 2)){ // 10 images
        printf("error loading MNIST data\n");
        return -1;
    }

    // ---- create, train, and test weights:
    // allocate memory to save results
    float zAnalysisData[gNumEpochs+1][gNumSimulations][gNumNoiseLevels];

    // run through six different noise levels
	  seed_randoms();
    for(int noiseLevel = 0; noiseLevel < gNumNoiseLevels; noiseLevel++){
      float zNoise = noiseLevel/10.0; // 0, .1, .2, .3, .4, .5
      for(int sim = 0; sim < gNumSimulations; sim++){

        // create weights connecting all input nodes to all output nodes
        float zWeights[gNumInNodes * gNumOutNodes];
        for (int i = 0; i < gNumInNodes * gNumOutNodes; i++){
            zWeights[i] = rand_weight();
            }

        // train and test the weights for the trial:
        for(int epoch = 0; epoch < gNumEpochs; epoch++){

            // test weights with learning off (zEta set to zero)
            float result;
            if(epoch == 0){ // get results once before any training
              result = runIt(zData, zWeights, zNoise, 0, sizeData);
              zAnalysisData[epoch][sim][noiseLevel] = result;
              }

            // train the weights, ignore testing during training
            result = runIt(zData, zWeights, zNoise, zEta, sizeData);
            zAnalysisData[epoch+1][sim][noiseLevel] = result;
        }
      }
    }

    // print results of learning averaged over number of trials
    printf("Image Noise = 0, Image Noise = .1, Image Noise = .2, ");
    printf("Image Noise = .3, Image Noise = .4, Image Noise = .5\n");
    for(int epoch = 0; epoch < gNumEpochs+1; epoch++){
      for(int noiseLevel = 0; noiseLevel < gNumNoiseLevels; noiseLevel++){
        float mean = 0;
        for(int sim = 0; sim < gNumSimulations; sim++){
            mean += zAnalysisData[epoch][sim][noiseLevel];
        }
        mean = mean/gNumSimulations;
        printf("%f, ", mean);
      }
      printf("\n");
    }

    free(zData);
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
float checkErr(float err[]){
    // find which idx of the output has the highest value
    float sumErr = 0.0;
    for(int i = 0; i < gNumOutNodes; i++){
        // two ways of estimating error
        sumErr += fabs(err[i]);
//        sumErr += pow(err[i], 2);
    }

    return sumErr;
}

/******************************************************************************/
float mySquash(float x)
{
    return 1/(1+(exp(-x)));
}

/******************************************************************************/
float runIt(mnist_data *pData, float pWeights[],
                                   float vNoise, float vEta, int vSizeData)
{
    // allocate memory
    int zIdx;
    int zInputVec[gNumInNodes];
    float zErr[gNumOutNodes];
    float zOutput[gNumOutNodes];
    float zTarget[gNumOutNodes];
    float zErrAcc = 0.0;

    // DO A TRAINING EPOCH (OR TESTING EPOCH IF ETA (learn rate)= 0)
    for(int cycle = 0; cycle < vSizeData; cycle++){

        // initialize output of all perceptrons to zero
        for (int i = 0; i<gNumOutNodes; i++) zOutput[i] = 0;

        // initialize training target (nine nodes off, target node on)
        for (int i = 0; i<gNumOutNodes; i++) zTarget[i] = 0;
        zTarget[pData[cycle].label] = 1;

        // translate image data to input vector w/ bias node
        get_input(zInputVec, pData, cycle, vNoise);
//        draw_input(zInputVec, pData[zIdx].label);

        // process each perceptron, given the input vector
        for(int j = 0; j < gNumOutNodes; j++){
            for(int i = 0; i < gNumInNodes; i++){
                zOutput[j] += zInputVec[i] * pWeights[(j*gNumInNodes)+i];
            }

            // squash the accumulated output for the perceptron:
            zOutput[j] = mySquash(zOutput[j]);

            // if in learning mode, update weights to the perceptron
            if(vEta != 0){//
                zErr[j] = zTarget[j] - zOutput[j];

                for(int i = 0; i < gNumInNodes; i++){
                    float weightChange = vEta * zErr[j] * zInputVec[i];
                    pWeights[(j*gNumInNodes)+i] += weightChange;
                }
//                zErrAcc += fabs(zErr[j]); // std way of getting error
            }
        }
        // two different ways of analyzing error, choose one:
//        zErrAcc += checkErr(zErr);
        zErrAcc += checkPrediction(zOutput, pData[cycle].label);
    }
    // return average error
    return (zErrAcc/(vSizeData * 1.0));
}
/*************************************************************************/
