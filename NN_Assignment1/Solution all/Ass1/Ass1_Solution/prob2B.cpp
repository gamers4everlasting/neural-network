/*******************************************************************************
*
*   Demo of NN Assignment 1 using multithreading
*   By Michael Brady
*     compile with: > g++ -o prob2B -std=c++11 prob2B.cpp -pthread
*
*   This example uses continuous input values rather than binary
*
*   Use learning rate as a command line argument, e.g. :
*   > test3 .001 > test.txt
*
*******************************************************************************/

#include <stdio.h>
#include <thread>
#include <stdlib.h> // needed for srand()
#include <time.h>   // needed for time keeping
#include <math.h>

#include "randlib.h"
#include "mnist/mnist.h"

#define gNumThreads 8         // number of threads to use
#define gNumInNodes 785       // (28*28)+1 = 785; // extra +1 for bias node
#define gNumOutNodes 10
#define gNumEpochs 20

#define VERBOSE 0

using namespace std;

int MULTITHREAD = 1; // multi-threading set to true by default
int SHUFFLE = 0;    // if shuffling the orer of training input

// some global variables to work with
mnist_data *gTrainData;                     // each image is 28x28 pixels
unsigned int gSizeTrainData;                // number if images in data set
mnist_data *gTestData;                      // each image is 28x28 pixels
unsigned int gSizeTestData;                 // number if images in data set
float gAnalysisData[gNumEpochs+1][gNumThreads];


/******************************************************************************/
float mySquash(float x)
{
    return 1.0/(1+(pow(M_E,-x)));
}

/******************************************************************************/
void shuffleList(int theList[], int sizeList)
{
    int swapVal;
    int swapLocation;
    for(int idx = 0; idx < sizeList; idx++){
      swapVal = theList[idx];
      swapLocation = floor(rand_frac() * sizeList);
      theList[idx] = theList[swapLocation];
      theList[swapLocation] = swapVal;
    }
}

/******************************************************************************/
void trainEpoch(float lRate, float zWeights[])
{
  float zInput[gNumInNodes];
  float zOutput[gNumOutNodes];
  float zErr[gNumOutNodes];
  float zTarget[gNumOutNodes];


  // in case user wants to randomize training order
  int trainingOrder[gSizeTrainData];
  int cycle;
  if(SHUFFLE){
    for(int cnt = 0; cnt<gSizeTrainData; cnt++){
      trainingOrder[cnt] = cnt;
    }
    shuffleList(trainingOrder, gSizeTrainData);
  }

  for(int cnt = 0; cnt < gSizeTrainData; cnt++){
    if(SHUFFLE) cycle = trainingOrder[cnt];
    else cycle = cnt;
      // get input vector - this is its own function in mnist.h
      // but now let's use continuous values for input instad of binarizing
      zInput[0] = 1; // for bias node
      for(int x = 0; x<28; x++){  // digit images are 28x28
        for(int y = 0; y<28; y++){
          zInput[(x*28)+y+1] = gTrainData[cycle].data[x][y];
        }
      }

      // get output target
      for (int j = 0; j<gNumOutNodes; j++) zTarget[j] = 0.0;
      zTarget[gTrainData[cycle].label] = 1.0;

      // update weights
      for (int j = 0; j<gNumOutNodes; j++) zOutput[j] = 0.0; // zero accumulator
      for(int j = 0; j < gNumOutNodes; j++){
        for(int i = 0; i < gNumInNodes; i++){
            zOutput[j] += zInput[i] * zWeights[(j*gNumInNodes)+i];
        }
        zOutput[j] = mySquash(zOutput[j]);

        zErr[j] = zTarget[j] - zOutput[j];

        for(int i = 0; i < gNumInNodes; i++){
            zWeights[(j*gNumInNodes)+i] += lRate * zErr[j] * zInput[i];
        }
      }
    } // end cycle
}

/******************************************************************************/
unsigned int testEpoch(float zWeights[])
{
  float zInput[gNumInNodes]; // could be double
  float zOutput[gNumOutNodes];
  float zErr[gNumOutNodes];
  float zTarget[gNumOutNodes];

  unsigned int errAcc = 0;

  for(int cycle = 0; cycle < gSizeTestData; cycle++){

    // get input vector - this is its own function in mnist.h
    // but now let's use continuous values for input instad of binarizing
      zInput[0] = 1; // for bias node
      for(int x = 0; x<28; x++){  // digit images are 28x28
        for(int y = 0; y<28; y++){
          zInput[(x*28)+y+1] = gTestData[cycle].data[x][y];
        }
      }

      // get output target
      for (int j = 0; j<gNumOutNodes; j++) zTarget[j] = 0.0;
      zTarget[gTestData[cycle].label] = 1.0;

      // simply get output
      for (int j = 0; j<gNumOutNodes; j++) zOutput[j] = 0.0;
      for(int j = 0; j < gNumOutNodes; j++){
        for(int i = 0; i < gNumInNodes; i++){
            zOutput[j] += zInput[i] * zWeights[(j*gNumInNodes)+i];
        }
        zOutput[j] = mySquash(zOutput[j]);
      }

      // test for correctness of the cycle - should have its own function
      float max = 0;
      int idx = 0;
      for(int j = 0; j < gNumOutNodes; j++){
        if(zOutput[j] > max){
          max = zOutput[j];
          idx = j;
        }
      }
      // if max value node is wrong prediciton, increase error
      if (idx != gTestData[cycle].label) errAcc += 1;
    } // end cycle

    return errAcc;
}

/******************************************************************************/
void runSim(int tid, float lRate)
{
  // create and initialize weights
  float zWeights[gNumInNodes * gNumOutNodes];
  for(int z=0; z<(gNumInNodes * gNumOutNodes); z++){
    zWeights[z] = rand_frac();
  }

  int numMistakes;
  for (int epoch = 0; epoch < gNumEpochs; epoch++){

    if(epoch == 0){ // test weights before training
      numMistakes = testEpoch(zWeights);
      gAnalysisData[epoch][tid] = (1.0 * numMistakes/gSizeTestData);
    }

    // then train and test weights
    trainEpoch(lRate, zWeights);
    numMistakes = testEpoch(zWeights); // ints to float by multiplying x 1.0
    gAnalysisData[epoch+1][tid] = (1.0 * numMistakes/gSizeTestData);

  } // end epoch
} // end main


/******************************************************************************/
void printResults(float lRate)
{
  printf("learn rate = %f\n", lRate);

  float mean;
  for(int epoch = 0; epoch < gNumEpochs+1; epoch++){
    mean = 0;
    for(int sim = 0; sim < gNumThreads; sim++){
      mean += gAnalysisData[epoch][sim]; // each thread a sim
    }
    mean = mean/gNumThreads;
    printf("%f\n", mean);
  }
}


/******************************************************************************/
int main(int argc, char* argv[])
{
  // initialize things
  seed_randoms();
  time_t start, end;
  thread zThreads[gNumThreads-1]; // maybe use this, maybe not..

  // get learning rate from the command line or use default of .001
  float lRate = .001;
  if(argc == 2){ // if an argument is provided, decides multithreading or not
    lRate = atof(argv[1]);
  }

  // load data as global so don't have to pass it around
  if (mnistLoad(&gTrainData, &gSizeTrainData, 0)){ // 60k images
    printf("error loading MNIST training data\n");
    return -1;
  }
  if (mnistLoad(&gTestData, &gSizeTestData, 1)){ // 10k images
    printf("error loading MNIST testing data\n");
    return -1;
  }

  if (MULTITHREAD){
    if (VERBOSE) printf("multithreading..\n");
    time(&start);

    // run threads on different CPUs
    for (int tid = 0; tid < gNumThreads-1; tid++){
      zThreads[tid] = thread(runSim, tid, lRate);
    }
    // run the last thread on the current cpu
    runSim(gNumThreads-1, lRate);

    // wait for threads to finish
    for (int tid = 0; tid < gNumThreads-1; tid++){
      zThreads[tid].join();
    }
    time(&end);
  }
  else{ // else doing it the slow way on a single core
    if (VERBOSE) printf("not multithreading..\n");
    time(&start);
    for(int tid = 0; tid < gNumThreads; tid++){
      runSim(tid, lRate);
    }
    time(&end);
  }

  printResults(lRate);

  // to inspect speed difference between threaded and unthreaded approach
  if (VERBOSE) printf("number of seconds used: %ld\n", end-start);
}
