#include <cuda_runtime.h>


__global__
void convReluKernel(const float* in, const float* w, const float* b,
                    float* out, int batchSize,
                    int inChannels, int inH, int inW,
                    int outChannels, int kH, int kW,
                    int outH, int outW,
                    int stride=1, int padding=0)
{
    // coordinate globali
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int batchIdx = blockIdx.z / outChannels;
    int filterIdx = blockIdx.z % outChannels;

    if (out_x >= outW || out_y >= outH || batchIdx >= batchSize) return;

    float result = 0.0f;

    for (int c = 0; c < inChannels; c++){
        for (int i = 0; i < kH; i++){
            for (int j = 0; j < kW; j++){
                int in_x = out_x * stride + j - padding;
                int in_y = out_y * stride + i - padding;
                if(in_x >= 0 && in_x < inW && in_y >= 0 && in_y < inH){
                    float inVal = in[batchIdx * (inChannels * inH * inW)
                                       + c * (inH * inW)
                                       + in_y * inW + in_x];
                    float wVal = w[filterIdx * (inChannels * kH * kW)
                                   + c * (kH * kW) + i * kW + j];
                    result += inVal * wVal;
                }
            }
        }
    }

    result += b[filterIdx];

    // ReLU
    if(result < 0.0f) result = 0.0f;

    out[batchIdx * (outChannels * outH * outW)
        + filterIdx * (outH * outW)
        + out_y * outW + out_x] = result;
}


__global__
void maxPoolFlattenKernel(const float* in, float* out,
                          int batchSize,
                          int inChannels, int inH, int inW,
                          int poolSize, int outH, int outW)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * inChannels * outH * outW;
    if(index >= total) return;

    int b = index / (inChannels * outH * outW);
    int rem = index % (inChannels * outH * outW);
    int c = rem / (outH * outW);
    int rem2 = rem % (outH * outW);
    int rOut = rem2 / outW;
    int cOut = rem2 % outW;

    int in_r = rOut * poolSize;
    int in_c = cOut * poolSize;

    int convStride = inW;

    float maxVal = -1e30f;
    for(int i=0; i<poolSize; i++){
        for(int j=0; j<poolSize; j++){
            int cur_r = in_r + i;
            int cur_c = in_c + j;
            if(cur_r < inH && cur_c < inW){
                float val = in[b * (inChannels * inH * inW) + c * (inH * inW) + cur_r * inW + cur_c];
                if(val > maxVal) maxVal = val;
            }
        }
    }

    out[index] = maxVal;
}


__global__
void fcForwardKernel(const float* in, const float* w, const float* b,
                     float* out,
                     int batchSize, int inFeatures, int outFeatures)
{
    const int TILE_SIZE = 16;

    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // batch
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // output feature

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    int numTiles = (inFeatures + TILE_SIZE - 1)/TILE_SIZE;

    for(int t=0; t<numTiles; t++){
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        if(row < batchSize && tiledCol < inFeatures)
            As[threadIdx.y][threadIdx.x] = in[row * inFeatures + tiledCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        int tiledRow = t * TILE_SIZE + threadIdx.y;
        if(tiledRow < inFeatures && col < outFeatures)
            Bs[threadIdx.y][threadIdx.x] = w[tiledRow * outFeatures + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        #pragma unroll
        for(int i=0; i<TILE_SIZE; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }

    if(row < batchSize && col < outFeatures)
        out[row * outFeatures + col] = sum + b[col];
}

__global__
void softmaxCrossEntropyKernel(const float* logits, const int* labels,
                               float* outLoss, float* outProb,
                               int batchSize, int numClasses)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= batchSize) return;

    int offset = i * numClasses;

    // 1) max logit (stabilità numerica)
    float maxLogit = -1e30f;
    for(int c = 0; c < numClasses; c++){
        float v = logits[offset + c];
        if(v > maxLogit) maxLogit = v;
    }

    // 2) exp + sum
    float sumExp = 0.0f;
    for(int c = 0; c < numClasses; c++){
        float ex = __expf(logits[offset + c] - maxLogit);
        outProb[offset + c] = ex;   // temporaneo
        sumExp += ex;
    }

    // 3) normalizzazione
    for(int c = 0; c < numClasses; c++)
        outProb[offset + c] /= sumExp;

    // 4) cross-entropy
    int lbl = labels[i];
    float p = outProb[offset + lbl];
    outLoss[i] = -logf(p + 1e-10f);
}
