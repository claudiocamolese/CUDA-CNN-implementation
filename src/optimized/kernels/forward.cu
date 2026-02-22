#include <cuda_runtime.h>

/**
 * @brief Performs the forward pass of a Convolution layer with Bias addition
 *        and ReLU activation fused into a single CUDA kernel.
 *
 * Supports configurable stride and zero-padding.
 *
 * Memory layout (row-major):
 *  - in  : [batchSize, inChannels, inH, inW]
 *  - w   : [outChannels, inChannels, kH, kW]
 *  - b   : [outChannels]
 *  - out : [batchSize, outChannels, outH, outW]
 *
 * -------------------------
 * Optimization Techniques Used
 * -------------------------
 *
 * 1. Kernel Fusion (Conv + Bias + ReLU):
 *    Convolution, bias addition, and ReLU activation are executed
 *    in a single kernel to:
 *      - Reduce global memory traffic
 *      - Avoid intermediate tensor writes
 *      - Improve cache locality
 *      - Reduce kernel launch overhead
 *
 * 2. Output-Element Parallelization:
 *    Each thread computes exactly one output element
 *    (one pixel for one filter and one batch sample).
 *    This eliminates race conditions and synchronization overhead.
 *
 * 3. 3D Grid Mapping:
 *    - blockIdx.x/y → spatial coordinates (outW, outH)
 *    - blockIdx.z   → encodes batch index and filter index
 *    This maximizes parallel coverage of output space.
 *
 * 4. Boundary Checks for Padding:
 *    Conditional checks avoid illegal memory accesses
 *    while implicitly implementing zero-padding.
 *
 *
 * @param in            Pointer to input tensor in device memory
 * @param w             Pointer to convolution filters in device memory
 * @param b             Pointer to bias vector in device memory
 * @param out           Pointer to output tensor in device memory
 * @param batchSize     Number of samples in the batch (B)
 * @param inChannels    Number of input channels (Cin)
 * @param inH           Input height
 * @param inW           Input width
 * @param outChannels   Number of output channels / filters (Cout)
 * @param kH            Kernel height
 * @param kW            Kernel width
 * @param outH          Output height
 * @param outW          Output width
 * @param stride        Convolution stride (default = 1)
 * @param padding       Zero-padding size applied to input (default = 0)
 *
 * @return void
 */
__global__ void ConvReluForward(const float* input, const float* w, const float* b, float* out, int batchSize, int inChannels, int inH, int inW, int outChannels, int kH, int kW, int outH, int outW, int stride=1, int padding=0)
{
    // compute global coordinates
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int batchIdx = blockIdx.z / outChannels;
    int filterIdx = blockIdx.z % outChannels;

    if (out_x >= outW || out_y >= outH || batchIdx >= batchSize) return;

    float result = 0.0f;

    for (int channel = 0; channel < inChannels; channel++){
        for (int i = 0; i < kH; i++){
            for (int j = 0; j < kW; j++){
                int in_x = out_x * stride + j - padding;
                int in_y = out_y * stride + i - padding;
                if(in_x >= 0 && in_x < inW && in_y >= 0 && in_y < inH){
                    float inVal = input[batchIdx * (inChannels * inH * inW) + channel * (inH * inW) + in_y * inW + in_x];
                    float wVal = w[filterIdx * (inChannels * kH * kW) + channel * (kH * kW) + i * kW + j];
                    result += inVal * wVal;
                }
            }
        }
    }

    result += b[filterIdx];

    // ReLU passage
    if(result < 0.0f) result = 0.0f;

    out[batchIdx * (outChannels * outH * outW) + filterIdx * (outH * outW) + out_y * outW + out_x] = result;
}




/**
 * @brief Performs Max Pooling followed by implicit Flatten in a single CUDA kernel.
 *
 * The output tensor is written in flattened linear memory layout:
 * index ∈ [0, batchSize * inChannels * outH * outW)
 *
 * Memory layout (row-major):
 *  - in  : [batchSize, inChannels, inH, inW]
 *  - out : [batchSize, inChannels, outH, outW] (flattened storage)
 *
 * Assumes non-overlapping pooling regions (stride = poolSize).
 *
 * -------------------------
 * Optimization Techniques Used
 * -------------------------
 *
 * 1. Kernel Fusion (MaxPool + Flatten):
 *    Pooling and flattening are combined into a single kernel.
 *    This:
 *      - Avoids an additional reshape kernel
 *      - Reduces global memory traffic
 *      - Eliminates intermediate writes
 *
 * 2. One-Thread-Per-Output Strategy:
 *    Each thread computes exactly one pooled output value.
 *    This removes synchronization requirements and race conditions.
 *
 * 3. Linear Index Decomposition:
 *    A single linear thread index is decomposed into:
 *      - batch index (b)
 *      - channel index (c)
 *      - pooled spatial coordinates (rOut, cOut)
 *    This simplifies parallel mapping and guarantees full coverage.
 *
 * 4. Bounds Checking:
 *    Ensures safe memory access when input dimensions
 *    are not perfectly divisible by poolSize.
 *
 *
 * @param in         Pointer to input tensor in device memory
 * @param out        Pointer to pooled + flattened output tensor
 * @param batchSize  Number of samples in the batch (B)
 * @param inChannels Number of input channels (Cin)
 * @param inH        Input height
 * @param inW        Input width
 * @param poolSize   Pooling window size (assumed square)
 * @param outH       Output height after pooling
 * @param outW       Output width after pooling
 *
 * @return void
 */
__global__ void MaxPoolFlattenForward(const float* in, float* out, int batchSize, int inChannels, int inH, int inW, int poolSize, int outH, int outW)
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

    //int convStride = inW;
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



/**
 * @brief Performs the forward pass of a Fully Connected (Dense) layer
 *        using tiled matrix multiplication with shared memory.
 *
 * Computes:
 *
 * out = in × w + b
 *
 * Where:
 *  - in  : [batchSize, inFeatures]
 *  - w   : [inFeatures, outFeatures]
 *  - b   : [outFeatures]
 *  - out : [batchSize, outFeatures]
 *
 * Each thread computes a single output element:
 * out[row, col]
 *
 * -------------------------
 * Optimization Techniques Used
 * -------------------------
 *
 * 1. Tiled Matrix Multiplication:
 *    The input matrix and weight matrix are divided into TILE_SIZE × TILE_SIZE submatrices (tiles).
 *    This reduces redundant global memory accesses.
 *
 * 2. Shared Memory Usage:
 *    Submatrices of 'in' and 'w' are loaded into shared memory
 *    (As and Bs) before computation.
 *    This:
 *      - Exploits fast on-chip memory
 *      - Improves memory bandwidth efficiency
 *      - Increases data reuse within a thread block
 *
 * 3. Cooperative Thread Loading:
 *    Threads within a block collaboratively load tiles into shared memory, maximizing memory coalescing.
 *
 * 4. Loop Tiling Over Input Features:
 *    The dot product is computed incrementally over tiles (numTiles), enabling scalability for large feature sizes.
 *
 * 5. Loop Unrolling:
 *    '#pragma unroll' reduces loop overhead and may improve instruction-level parallelism.
 *
 * 6. One-Thread-Per-Output-Element Strategy:
 *    Each thread computes exactly one output element, eliminating race conditions and synchronization complexity beyond tile synchronization.
 *
 * 7. Boundary Handling:
 *    Conditional checks prevent out-of-bounds accesses when dimensions are not multiples of TILE_SIZE.
 *
 *
 * @param in            Pointer to input tensor in device memory
 * @param w             Pointer to weight matrix in device memory
 * @param b             Pointer to bias vector in device memory
 * @param out           Pointer to output tensor in device memory
 * @param batchSize     Number of samples in the batch (B)
 * @param inFeatures    Number of input features (Fin)
 * @param outFeatures   Number of output features (Fout)
 *
 * @return void
 */
__global__ void FCForward(const float* in, const float* w, const float* b, float* out, int batchSize, int inFeatures, int outFeatures)
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





/**
 * @brief Computes Softmax probabilities and Cross-Entropy loss (fused)
 *        for a batch of logits.
 *
 * Outputs:
 *  - outProb : softmax probabilities [batchSize, numClasses]
 *  - outLoss : per-sample cross-entropy loss [batchSize]
 *
 * Memory layout (row-major):
 *  - logits  : [batchSize, numClasses]
 *  - labels  : [batchSize]
 *  - outProb : [batchSize, numClasses]
 *  - outLoss : [batchSize]
 *
 * -------------------------
 * Optimization Techniques Used
 * -------------------------
 *
 * 1. Kernel Fusion (Softmax + Cross-Entropy):
 *    Softmax computation and cross-entropy loss are performed
 *    in a single kernel. This:
 *      - Eliminates intermediate global memory writes
 *      - Reduces memory bandwidth usage
 *      - Reduces kernel launch overhead
 *
 * 2. One-Thread-Per-Sample Strategy:
 *    Each thread processes one batch element.
 *    This avoids inter-thread synchronization and ensures
 *    independence across samples.
 *
 * 3. Numerical Stability Trick:
 *    The maximum logit is subtracted before exponentiation:
 *        exp(logit - maxLogit)
 *    This prevents overflow and improves floating-point stability.
 *
 * 4. Fast Approximate Exponential:
 *    Uses __expf() instead of expf() for faster computation on GPU, trading slight precision loss for performance.
 *
 * 5. In-Place Temporary Storage:
 *    The outProb buffer temporarily stores unnormalized exponentials before normalization, avoiding extra buffers.
 *
 * 6. Log Epsilon Protection:
 *    A small constant (1e-10f) is added inside log() to prevent log(0) and improve numerical robustness.
 *
 *
 * @param logits      Pointer to input logits in device memory
 * @param labels      Pointer to ground-truth class indices
 * @param outLoss     Pointer to per-sample loss output
 * @param outProb     Pointer to softmax probabilities output
 * @param batchSize   Number of samples in the batch (B)
 * @param numClasses  Number of classes (C)
 *
 * @return void
 */
__global__ void SoftmaxForward(const float* logits, const int* labels, float* outLoss, float* outProb, int batchSize, int numClasses)
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
