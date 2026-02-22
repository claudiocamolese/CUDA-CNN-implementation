#include <cuda_runtime.h>


__global__ void softmaxCrossEntropyBackwardKernel(float* gradLogits, const float* prob, const int* labels, int batchSize, int numClasses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batchSize * numClasses) return;

    int sampleIdx = idx / numClasses;
    int c = idx % numClasses;
    int lbl = labels[sampleIdx];
    float y = (c == lbl) ? 1.0f : 0.0f;
    gradLogits[idx] = prob[idx] - y;
}


__global__ void fcBackwardGradParamKernel(const float* gradOut, const float* in, float* gradW, float* gradB, int batchSize, int inFeatures, int outFeatures)
{
    const int TILE_SIZE = 16;
    int totalW = inFeatures * outFeatures;
    int totalParams = totalW + outFeatures;
    int paramIdx = blockIdx.x * TILE_SIZE + threadIdx.y;
    if(paramIdx >= totalParams) return;

    float sum = 0.0f;
    bool isWeight = (paramIdx < totalW);
    int k=0, c=0;
    if(isWeight){
        k = paramIdx / outFeatures;
        c = paramIdx % outFeatures;
    } else {
        c = paramIdx - totalW;
    }

    int stride = blockDim.x;
    for(int b=threadIdx.x; b<batchSize; b+=stride){
        if(isWeight) sum += in[b * inFeatures + k] * gradOut[b * outFeatures + c];
        else sum += gradOut[b * outFeatures + c];
    }

    __shared__ float sdata[32][TILE_SIZE];
    sdata[threadIdx.x][threadIdx.y] = sum;
    __syncthreads();

    for(int s=blockDim.x/2; s>0; s>>=1){
        if(threadIdx.x < s) sdata[threadIdx.x][threadIdx.y] += sdata[threadIdx.x + s][threadIdx.y];
        __syncthreads();
    }

    if(threadIdx.x == 0){
        if(isWeight) gradW[paramIdx] = sdata[0][threadIdx.y];
        else gradB[paramIdx - totalW] = sdata[0][threadIdx.y];
    }
}


__global__ void fcBackwardGradInKernel(const float* gradOut, const float* w, float* gradIn, int batchSize, int inFeatures, int outFeatures)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batchSize * inFeatures) return;

    int b = idx / inFeatures;
    int k = idx % inFeatures;
    float sumVal = 0.0f;
    int baseGrad = b * outFeatures;
    int baseW = k * outFeatures;

    #pragma unroll
    for(int c=0; c<outFeatures; c++)
        sumVal = fmaf(gradOut[baseGrad + c], w[baseW + c], sumVal);

    gradIn[idx] = sumVal;
}


__global__ void reluBackwardKernel(const float* gradOut, const float* x, float* gradIn, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        float x_val = __ldg(&x[i]);
        gradIn[i] = gradOut[i] * (x_val > 0.0f);
    }
}


__global__ void maxPoolBackwardKernel(const float* convOut, const float* gradFlat, float* gradConvOut, int batchSize, int inChannels, int inH, int inW, int poolSize, int outH, int outW)
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
    int base = b * (inChannels * inH * inW) + c * inH * inW + in_r * convStride + in_c;

    float maxVal = convOut[base];
    int maxIdx = 0;
    for(int i=0; i<poolSize; i++){
        for(int j=0; j<poolSize; j++){
            int cur_r = in_r + i;
            int cur_c = in_c + j;
            if(cur_r<inH && cur_c<inW){
                float val = convOut[b * (inChannels * inH * inW) + c * inH * inW + cur_r * inW + cur_c];
                if(val > maxVal){
                    maxVal = val;
                    maxIdx = i*poolSize+j;
                }
            }
        }
    }

    int write_r = in_r + maxIdx / poolSize;
    int write_c = in_c + maxIdx % poolSize;
    gradConvOut[b * (inChannels * inH * inW) + c * inH * inW + write_r * inW + write_c] = gradFlat[index];
}


__global__ void convBackwardWeightKernel(const float* in, const float* gradConvOut, float* gradW, float* gradB, int batchSize, int inChannels, int inH, int inW, int outChannels, int kH, int kW, int outH, int outW)
{
    int totalW = outChannels * inChannels * kH * kW;
    int totalParams = totalW + outChannels;
    int paramIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(paramIdx >= totalParams) return;

    float sum = 0.0f;
    int warpSize = 32;

    if(paramIdx < totalW){
        int f = paramIdx / (inChannels * kH * kW);
        int rem = paramIdx % (inChannels * kH * kW);
        int c = rem / (kH * kW);
        int kr = (rem % (kH * kW)) / kW;
        int kc = rem % kW;

        int N = batchSize * outH * outW;
        for(int i=threadIdx.x; i<N; i+=warpSize){
            int b = i / (outH * outW);
            int s = i % (outH * outW);
            int orow = s / outW;
            int ocol = s % outW;

            float g = gradConvOut[b * (outChannels * outH * outW) + f * (outH * outW) + orow * outW + ocol];
            float inp = in[b * (inChannels * inH * inW) + c * (inH * inW) + (orow + kr) * inW + (ocol + kc)];
            sum += inp * g;
        }
    } else {
        int f = paramIdx - totalW;
        int N = batchSize * outH * outW;
        for(int i=threadIdx.x; i<N; i+=warpSize){
            int b = i / (outH * outW);
            int s = i % (outH * outW);
            int orow = s / outW;
            int ocol = s % outW;
            sum += gradConvOut[b * (outChannels * outH * outW) + f * (outH * outW) + orow * outW + ocol];
        }
    }

    unsigned int mask = 0xffffffff;
    for(int offset = warpSize/2; offset>0; offset/=2)
        sum += __shfl_down_sync(mask, sum, offset);

    if(threadIdx.x==0){
        if(paramIdx<totalW) gradW[paramIdx]=sum;
        else gradB[paramIdx-totalW]=sum;
    }
}


__global__ void convBackwardInputKernel(const float* gradOut, const float* w, float* gradIn, int batchSize, int inChannels, int inH, int inW, int outChannels, int kH, int kW, int outH, int outW)
{
    int b = blockIdx.x;
    if(b >= batchSize) return;

    int c = threadIdx.z;   // input channel
    int r = threadIdx.y;
    int col = threadIdx.x;

    if(c >= inChannels || r >= inH || col >= inW) return;

    float sum = 0.0f;

    for(int f = 0; f < outChannels; f++){
        for(int kr = 0; kr < kH; kr++){
            for(int kc = 0; kc < kW; kc++){
                int orow = r - kr;
                int ocol = col - kc;
                if(orow >= 0 && orow < outH && ocol >= 0 && ocol < outW){
                    float g =
                        gradOut[b * (outChannels * outH * outW)
                        + f * (outH * outW)
                        + orow * outW + ocol];

                    float wv =
                        w[f * (inChannels * kH * kW)
                        + c * (kH * kW)
                        + kr * kW + kc];

                    sum = fmaf(g, wv, sum);
                }
            }
        }
    }

    gradIn[b * (inChannels * inH * inW) + c * (inH * inW) + r * inW + col] = sum;
}


__global__ void SGDBackward(float* param, const float* grad, float lr, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i< n) param[i] -= lr * grad[i];
}
