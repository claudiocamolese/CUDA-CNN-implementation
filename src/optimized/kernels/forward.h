#pragma once
#include <cuda_runtime.h>

__global__
void convReluKernel(const float* in, const float* w, const float* b,
                    float* out, int batchSize,
                    int inChannels, int inH, int inW,
                    int outChannels, int kH, int kW,
                    int outH, int outW,
                    int stride=1, int padding=0);

__global__
void maxPoolFlattenKernel(const float* in, float* out,
                          int batchSize,
                          int inChannels, int inH, int inW,
                          int poolSize, int outH, int outW);

__global__
void fcForwardKernel(const float* in, const float* w, const float* b,
                     float* out,
                     int batchSize, int inFeatures, int outFeatures);

__global__
void softmaxCrossEntropyKernel(const float* logits, const int* labels,
                               float* outLoss, float* outProb,
                               int batchSize, int numClasses);
