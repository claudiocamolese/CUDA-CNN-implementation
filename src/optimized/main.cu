/*
    This code is an optimized implementation of a deep convolutional neural network.

    It is implemented entirely from scratch without using any deep learning libraries.
    The network consists of two convolutional layers with ReLU activations, followed by
    a max pooling layer, a flatten operation, a fully connected layer and a softmax
    output for classification.

    The forward pass includes convolution, activation, pooling, flattening, and
    fully connected computations, while the backward pass computes gradients for
    all learnable parameters and propagates them through the network to update
    weights and biases using stochastic gradient descent.
 */

#include "config.h"
#include "utils/load_data.h"
#include "utils/printing.h"
#include "kernels/forward.h"
#include "kernels/backward.h"
#include <cuda_runtime.h>
#include <iostream>


/**
 * @brief Checks the result of a CUDA runtime call and exits on error.
 *
 * @param err The cudaError_t returned by a CUDA API call.
 * @param msg Optional custom message describing the context of the call.
 */

inline void CudaCheck(cudaError_t err, const char* msg = "") {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s (%s)\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    
    srand(21); // for reproducibility

    /*
        Allocate space for train and test set, images are 28x28  
    */
    float* h_trainImages = (float*)malloc(TRAIN_IMAGES * IMAGE_ROWS * IMAGE_COLS * sizeof(float));
    float* h_testImages  = (float*)malloc(TEST_IMAGES  * IMAGE_ROWS * IMAGE_COLS * sizeof(float));
    int*   h_trainLabels = (int*)malloc(TRAIN_IMAGES * sizeof(int));
    int*   h_testLabels  = (int*)malloc(TEST_IMAGES  * sizeof(int));

    if (!h_trainImages || !h_testImages || !h_trainLabels || !h_testLabels) {
        printf("Memory allocation failed\n");
        return 1;
    }

    /* 
        Load the dataset, mnist is the default one.
    */
    std::string dataset = "mnist"; // default

    if (argc == 2) {
        dataset = argv[1];
    }

    std::string train_images = "../datasets/" + dataset + "/train-images.idx3-ubyte";
    std::string test_images ="../datasets/" + dataset + "/t10k-images.idx3-ubyte";
    std::string train_labels = "../datasets/" + dataset + "/train-labels.idx1-ubyte";
    std::string test_labels = "../datasets/" + dataset + "/t10k-labels.idx1-ubyte";

    load_images(train_images.c_str(), h_trainImages, TRAIN_IMAGES);
    load_images(test_images.c_str(), h_testImages, TEST_IMAGES);
    load_labels(train_labels.c_str(), h_trainLabels, TRAIN_IMAGES);
    load_labels(test_labels.c_str(), h_testLabels, TEST_IMAGES);

    printf("Network --> (CONV(1, %d, %d), RELU) + (CONV(%d, %d, %d), RELU) + MaxPool(%d) + Flatten + FC + SoftMax\n", FIRST_OUTPUT_CHANNELS, FILTER_SIZE, FIRST_OUTPUT_CHANNELS, SECOND_OUTPUT_CHANNELS, FILTER_SIZE, POOL_SIZE);
    printf("Epochs: %d, Learning rate: %.2f, Batch size: %d\n", EPOCHS, LEARNING_RATE, BATCH_SIZE);
    printf("Block Size: %d\n", BLOCK_SIZE);


    // --------------------
    //  Initialization
    // --------------------
    
    // Convolutional layers
    float *d_conv1W, *d_conv1B, *d_conv1Out;
    float *d_conv2W, *d_conv2B, *d_conv2Out;
    
    // Pooling
    float *d_poolOut;
    int   *d_poolIdx;
    
    // Flatten + Fully Connected
    float *d_flat, *d_fcW, *d_fcB, *d_fcOut;
    
    // Prob, loss, gradients
    float *d_prob, *d_loss;
    float *d_grad_fcW, *d_grad_fcB;
    float *d_grad_fcOut, *d_grad_flat;
    float *d_grad_conv1Out, *d_grad_conv1W, *d_grad_conv1B;
    float *d_grad_conv2Out, *d_grad_conv2W, *d_grad_conv2B;
    float *d_grad_poolOut, *d_grad_in;

    /* 
    Triple-buffered batch data

    Three separate buffers are allocated on both the CPU (pinned memory)
    and the GPU for batch data (images and labels). The goal is to allow asynchronous 
    data transfers from CPU to GPU using CUDA streams, so that while the GPU processes 
    one batch, the CPU can prepare the next batch and copy it into an available buffer, 
    avoiding idle time.

    - d_trainImages[i] and d_labels[i]: pointers to the buffers in device (GPU) memory
    - h_pinnedImages[i] and h_pinnedLabels[i]: pointers to the buffers in pinned CPU memory
    - stream[i]: CUDA stream associated with each buffer, allowing asynchronous memcpy 
      and kernel execution in parallel without blocking the GPU.

    NUM_BUFFERS = 3 (triple buffering) is used to maximize overlap between data transfers 
    and computation, improving the training pipeline efficiency. Higher NUM_BUFFERS is not useful.
*/

    size_t imageBytesPerBatch = BATCH_SIZE * IMAGE_ROWS * IMAGE_COLS * sizeof(float);
    float* d_trainImages[NUM_BUFFERS];
    int*   d_labels[NUM_BUFFERS];

    for (int i = 0; i < NUM_BUFFERS; i++){
        CudaCheck(cudaMalloc(&d_trainImages[i], imageBytesPerBatch));
        CudaCheck(cudaMalloc(&d_labels[i], BATCH_SIZE * sizeof(int)));
    }

    cudaStream_t stream[NUM_BUFFERS];
    float* h_pinnedImages[NUM_BUFFERS];
    int*   h_pinnedLabels[NUM_BUFFERS];

    for (int i = 0; i < NUM_BUFFERS; i++){
        CudaCheck(cudaStreamCreate(&stream[i]));
        CudaCheck(cudaMallocHost((void**)&h_pinnedImages[i], imageBytesPerBatch));
        CudaCheck(cudaMallocHost((void**)&h_pinnedLabels[i], BATCH_SIZE * sizeof(int)));
    }

    // Dimensions
    int conv1W_size = FIRST_OUTPUT_CHANNELS  * FIRST_INPUT_CHANNELS  * FILTER_SIZE * FILTER_SIZE;
    int conv2W_size = SECOND_OUTPUT_CHANNELS * SECOND_INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE;   

    // -------------------------
    // CUDA memory allocation
    // -------------------------
    CudaCheck(cudaMalloc(&d_conv1W, conv1W_size * sizeof(float)));
    CudaCheck(cudaMalloc(&d_conv1B, FIRST_OUTPUT_CHANNELS * sizeof(float)));
    CudaCheck(cudaMalloc(&d_conv1Out, BATCH_SIZE * FIRST_OUTPUT_CHANNELS * CONV1_OUT_ROWS * CONV1_OUT_COLS * sizeof(float)));

    CudaCheck(cudaMalloc(&d_conv2W, conv2W_size * sizeof(float)));
    CudaCheck(cudaMalloc(&d_conv2B, SECOND_OUTPUT_CHANNELS * sizeof(float)));
    CudaCheck(cudaMalloc(&d_conv2Out, BATCH_SIZE * SECOND_OUTPUT_CHANNELS * CONV2_OUT_ROWS * CONV2_OUT_COLS * sizeof(float)));

    CudaCheck(cudaMalloc(&d_poolOut, BATCH_SIZE * SECOND_OUTPUT_CHANNELS * POOL_OUT_ROWS * POOL_OUT_COLS * sizeof(float)));
    CudaCheck(cudaMalloc(&d_poolIdx, BATCH_SIZE * SECOND_OUTPUT_CHANNELS * POOL_OUT_ROWS * POOL_OUT_COLS * sizeof(int)));

    CudaCheck(cudaMalloc(&d_flat, BATCH_SIZE * FLATTEN_SIZE * sizeof(float)));
    CudaCheck(cudaMalloc(&d_fcW, FLATTEN_SIZE * NUM_CLASSES * sizeof(float)));
    CudaCheck(cudaMalloc(&d_fcB, NUM_CLASSES * sizeof(float)));
    CudaCheck(cudaMalloc(&d_fcOut, BATCH_SIZE * NUM_CLASSES * sizeof(float)));

    CudaCheck(cudaMalloc(&d_prob, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    CudaCheck(cudaMalloc(&d_loss, BATCH_SIZE * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_fcOut, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_flat, BATCH_SIZE * FLATTEN_SIZE * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_fcW, FLATTEN_SIZE * NUM_CLASSES * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_fcB, NUM_CLASSES * sizeof(float)));

    CudaCheck(cudaMalloc(&d_grad_conv1Out, BATCH_SIZE * FIRST_OUTPUT_CHANNELS * CONV1_OUT_ROWS * CONV1_OUT_COLS * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_conv1W, conv1W_size * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_conv1B, FIRST_OUTPUT_CHANNELS * sizeof(float)));

    CudaCheck(cudaMalloc(&d_grad_conv2Out, BATCH_SIZE * SECOND_OUTPUT_CHANNELS * CONV2_OUT_ROWS * CONV2_OUT_COLS * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_conv2W, conv2W_size * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_conv2B, SECOND_OUTPUT_CHANNELS * sizeof(float)));

    CudaCheck(cudaMalloc(&d_grad_poolOut, BATCH_SIZE * SECOND_OUTPUT_CHANNELS * POOL_OUT_ROWS * POOL_OUT_COLS * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_in, BATCH_SIZE * IMAGE_ROWS * IMAGE_COLS * sizeof(float)));

    
    /* ------------------------------------------------------------
        He weights initialization on host and copied to device
        ------------------------------------------------------------
        
        He initialization, is better suited for layers that use ReLU 
        activation functions since it mitigates the exploding gradient issue.
        The layer weights are initialized in the range [-limit, +limit] while the bias are initialized to 0.
        The limit is W ~ U(- sqrt(6/n), sqrt(6/n)) where n is the number of input neurons to the layer.
    */

    // Initilization of the first Conv layer (Conv1) 
    float* h_conv1W = (float*)malloc(conv1W_size * sizeof(float)); // layer weights
    float* h_conv1B = (float*)malloc(FIRST_OUTPUT_CHANNELS * sizeof(float)); // bias weights
    int fan_in_conv1 = FIRST_INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE; // total input number for each neuron, use to scale the weights
    auto he_rand_conv1 = [fan_in_conv1]() {
        float limit = sqrtf(6.0f / fan_in_conv1); // He initilization
        return limit * (2.0f * ((float)rand() / RAND_MAX) - 1.0f); // casual number in the range [-limit, + limit]
    };
    for (int i = 0; i < conv1W_size; i++) h_conv1W[i] = he_rand_conv1();
    for (int i = 0; i < FIRST_OUTPUT_CHANNELS; i++) h_conv1B[i] = 0.0f;
    CudaCheck(cudaMemcpy(d_conv1W, h_conv1W, conv1W_size * sizeof(float), cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(d_conv1B, h_conv1B, FIRST_OUTPUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice));
    free(h_conv1W); 
    free(h_conv1B);

    // Initilization of the second Conv layer (Conv2)
    float* h_conv2W = (float*)malloc(conv2W_size * sizeof(float));
    float* h_conv2B = (float*)malloc(SECOND_OUTPUT_CHANNELS * sizeof(float));
    int fan_in_conv2 = SECOND_INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE;
    auto he_rand_conv2 = [fan_in_conv2]() {
        float limit = sqrtf(6.0f / fan_in_conv2);
        return limit * (2.0f * ((float)rand() / RAND_MAX) - 1.0f);
    };
    for (int i = 0; i < conv2W_size; i++) h_conv2W[i] = he_rand_conv2();
    for (int i = 0; i < SECOND_OUTPUT_CHANNELS; i++) h_conv2B[i] = 0.0f;
    CudaCheck(cudaMemcpy(d_conv2W, h_conv2W, conv2W_size * sizeof(float), cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(d_conv2B, h_conv2B, SECOND_OUTPUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice));
    free(h_conv2W); 
    free(h_conv2B);

    // Initilization of the FullyConnected layer (FC)
    float* h_fcW = (float*)malloc(FLATTEN_SIZE * NUM_CLASSES * sizeof(float));
    float* h_fcB = (float*)malloc(NUM_CLASSES * sizeof(float));
    int fan_in_fc = FLATTEN_SIZE;
    auto he_rand_fc = [fan_in_fc]() {
        float limit = sqrtf(6.0f / fan_in_fc);
        return limit * (2.0f * ((float)rand() / RAND_MAX) - 1.0f);
    };
    for (int i = 0; i < FLATTEN_SIZE * NUM_CLASSES; i++) h_fcW[i] = he_rand_fc();
    for (int i = 0; i < NUM_CLASSES; i++) h_fcB[i] = 0.0f;
    CudaCheck(cudaMemcpy(d_fcW, h_fcW, FLATTEN_SIZE * NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(d_fcB, h_fcB, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice));
    free(h_fcW); free(h_fcB);

    // Record time for training and testing
    cudaEvent_t startEvent, stopEvent;
    CudaCheck(cudaEventCreate(&startEvent));
    CudaCheck(cudaEventCreate(&stopEvent));
    CudaCheck(cudaEventRecord(startEvent, 0));

    printf("Starting training for %d epochs:\n", EPOCHS);

    for (int epoch= 0; epoch< EPOCHS; epoch++){
        
        float epoch_loss = 0.0f;
        int uploadedBuffers = (NUM_BATCHES < NUM_BUFFERS) ? NUM_BATCHES : NUM_BUFFERS; // correctly manage last batches

        for(int i=0; i < uploadedBuffers ; i++){
            
            int BatchSize = i * BATCH_SIZE * (IMAGE_ROWS * IMAGE_COLS);
            
            memcpy(h_pinnedImages[i], &h_trainImages[BatchSize], imageBytesPerBatch);
            memcpy(h_pinnedLabels[i], &h_trainLabels[i * BATCH_SIZE], BATCH_SIZE * sizeof(int));
            
            // Copy data without blocking the CPU, using the straming CUDA
            CudaCheck(cudaMemcpyAsync(d_trainImages[i], h_pinnedImages[i], imageBytesPerBatch, cudaMemcpyHostToDevice, stream[i]));
            CudaCheck(cudaMemcpyAsync(d_labels[i], h_pinnedLabels[i], BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream[i]));
        }
        
        for(int batch = 0; batch < NUM_BATCHES; batch++){

            print_progress(batch + 1, NUM_BATCHES, epoch, EPOCHS);

            int CurrentBuffer = batch % NUM_BUFFERS;
            int NextBuffer = (batch + 1) % NUM_BUFFERS;

            // Before using the current buffer data, must wait that the asynchronize batch copy on the GPU is complete.
            CudaCheck(cudaStreamSynchronize(stream[CurrentBuffer]));

            // CONV1 + RELU
            {
            dim3 blockDim(16, 16, 1);
            dim3 gridDim((CONV1_OUT_COLS + 16 - 1)/16,
                        (CONV1_OUT_ROWS + 16 - 1)/16,
                        BATCH_SIZE * FIRST_OUTPUT_CHANNELS);

            size_t SharedMemorySize = (16 + FILTER_SIZE - 1) * (16 + FILTER_SIZE - 1) * sizeof(float); // to reduce global memory acess

            convReluKernel<<<gridDim, blockDim, SharedMemorySize, stream[CurrentBuffer]>>>(
                d_trainImages[CurrentBuffer], d_conv1W, d_conv1B, d_conv1Out, 
                BATCH_SIZE, FIRST_INPUT_CHANNELS, IMAGE_ROWS, IMAGE_COLS,
                FIRST_OUTPUT_CHANNELS, FILTER_SIZE, FILTER_SIZE,
                CONV1_OUT_ROWS, CONV1_OUT_COLS);
            }

            // Conv2 + ReLU
            {
            dim3 blockDim(16, 16, 1);
            dim3 gridDim((CONV2_OUT_COLS + 16 - 1)/16,
                        (CONV2_OUT_ROWS + 16 - 1)/16,
                        BATCH_SIZE * SECOND_OUTPUT_CHANNELS);

            size_t SharedMemorySize = (16 + FILTER_SIZE - 1) * (16 + FILTER_SIZE - 1) * sizeof(float);

            convReluKernel<<<gridDim, blockDim, SharedMemorySize, stream[CurrentBuffer]>>>(
                d_conv1Out, d_conv2W, d_conv2B, d_conv2Out, 
                BATCH_SIZE, SECOND_INPUT_CHANNELS, CONV1_OUT_ROWS, CONV1_OUT_COLS,
                SECOND_OUTPUT_CHANNELS, FILTER_SIZE, FILTER_SIZE,
                CONV2_OUT_ROWS, CONV2_OUT_COLS);
            }


            // Max Pooling + Flatten
            {
            int totalFlatten = BATCH_SIZE * SECOND_OUTPUT_CHANNELS * POOL_OUT_ROWS * POOL_OUT_COLS;
            int grid = (totalFlatten + BLOCK_SIZE - 1) / BLOCK_SIZE;

            maxPoolFlattenKernel<<<grid, BLOCK_SIZE, 0, stream[CurrentBuffer]>>>(
                d_conv2Out, d_flat,
                BATCH_SIZE,SECOND_OUTPUT_CHANNELS, CONV2_OUT_ROWS, CONV2_OUT_COLS,
                POOL_SIZE, POOL_OUT_ROWS, POOL_OUT_COLS);
            }


            // Fully Connected forward (tiled GEMM)
            {
            dim3 fcGrid((NUM_CLASSES + 16 - 1)/16, (BATCH_SIZE + 16 - 1)/16);
            dim3 fcBlock(16, 16, 1);
            int sharedMemBytes = 2 * (16*16) * sizeof(float);

            fcForwardKernel<<<fcGrid, fcBlock, sharedMemBytes, stream[CurrentBuffer]>>>(
                d_flat, d_fcW, d_fcB, d_fcOut,
                BATCH_SIZE, FLATTEN_SIZE, NUM_CLASSES);
            }

            // --------------------
            // Softmax + Cross-Entropy Loss
            // --------------------
            {
            int grid = (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t sharedMemBytes = NUM_CLASSES * sizeof(float); // per expVals

            softmaxCrossEntropyKernel<<<grid, BLOCK_SIZE, sharedMemBytes, stream[CurrentBuffer]>>>(
                d_fcOut, d_labels[CurrentBuffer], d_loss, d_prob,
                BATCH_SIZE, NUM_CLASSES
            );
            }
            // ============================================================================
            // COMPLETE BACKWARD PASS — insert inside training loop AFTER softmax forward
            // ============================================================================

            // ---------------------------------------------------------------------------
            // 1) Softmax + Cross-Entropy backward  -> d_grad_fcOut
            // ---------------------------------------------------------------------------
            {
                int total = BATCH_SIZE * NUM_CLASSES;
                int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

                softmaxCrossEntropyBackwardKernel<<<grid, BLOCK_SIZE, 0, stream[CurrentBuffer]>>>(
                    d_grad_fcOut,
                    d_prob,
                    d_labels[CurrentBuffer],
                    BATCH_SIZE,
                    NUM_CLASSES
                );
            }

            // ---------------------------------------------------------------------------
            // 2) Fully Connected backward (grad W, B)
            // ---------------------------------------------------------------------------
            {
                int totalParams = FLATTEN_SIZE * NUM_CLASSES + NUM_CLASSES;
                dim3 blockFC(32, 16);
                dim3 gridFC((totalParams + 15) / 16);

                fcBackwardGradParamKernel<<<gridFC, blockFC, 0, stream[CurrentBuffer]>>>(
                    d_grad_fcOut,
                    d_flat,
                    d_grad_fcW,
                    d_grad_fcB,
                    BATCH_SIZE,
                    FLATTEN_SIZE,
                    NUM_CLASSES
                );
            }

            // ---------------------------------------------------------------------------
            // 3) Fully Connected backward (grad input -> d_grad_flat)
            // ---------------------------------------------------------------------------
            {
                int total = BATCH_SIZE * FLATTEN_SIZE;
                int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

                fcBackwardGradInKernel<<<grid, BLOCK_SIZE, 0, stream[CurrentBuffer]>>>(
                    d_grad_fcOut,
                    d_fcW,
                    d_grad_flat,
                    BATCH_SIZE,
                    FLATTEN_SIZE,
                    NUM_CLASSES
                );
            }

            // ---------------------------------------------------------------------------
            // 4) MaxPool backward -> d_grad_conv2Out
            // ---------------------------------------------------------------------------
            {
                cudaMemsetAsync(
                    d_grad_conv2Out,
                    0,
                    BATCH_SIZE * SECOND_OUTPUT_CHANNELS *
                    CONV2_OUT_ROWS * CONV2_OUT_COLS * sizeof(float),
                    stream[CurrentBuffer]
                );

                int total = BATCH_SIZE * SECOND_OUTPUT_CHANNELS *
                            POOL_OUT_ROWS * POOL_OUT_COLS;
                int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

                maxPoolBackwardKernel<<<grid, BLOCK_SIZE, 0, stream[CurrentBuffer]>>>(
                    d_conv2Out,
                    d_grad_flat,
                    d_grad_conv2Out,
                    BATCH_SIZE,
                    SECOND_OUTPUT_CHANNELS,
                    CONV2_OUT_ROWS,
                    CONV2_OUT_COLS,
                    POOL_SIZE,
                    POOL_OUT_ROWS,
                    POOL_OUT_COLS
                );
            }

            // ---------------------------------------------------------------------------
            // 5) ReLU backward (Conv2)
            // ---------------------------------------------------------------------------
            {
                int total = BATCH_SIZE * SECOND_OUTPUT_CHANNELS *
                            CONV2_OUT_ROWS * CONV2_OUT_COLS;
                int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

                reluBackwardKernel<<<grid, BLOCK_SIZE, 0, stream[CurrentBuffer]>>>(
                    d_grad_conv2Out,
                    d_conv2Out,
                    d_grad_conv2Out,
                    total
                );
            }

            // ---------------------------------------------------------------------------
            // 6) Conv2 backward (grad W, B)
            // ---------------------------------------------------------------------------
            {
                int totalParams =
                    SECOND_OUTPUT_CHANNELS * SECOND_INPUT_CHANNELS *
                    FILTER_SIZE * FILTER_SIZE +
                    SECOND_OUTPUT_CHANNELS;

                int grid = (totalParams + 31) / 32;

                convBackwardWeightKernel<<<grid, 32, 0, stream[CurrentBuffer]>>>(
                    d_conv1Out,
                    d_grad_conv2Out,
                    d_grad_conv2W,
                    d_grad_conv2B,
                    BATCH_SIZE,
                    SECOND_INPUT_CHANNELS,
                    CONV1_OUT_ROWS,
                    CONV1_OUT_COLS,
                    SECOND_OUTPUT_CHANNELS,
                    FILTER_SIZE,
                    FILTER_SIZE,
                    CONV2_OUT_ROWS,
                    CONV2_OUT_COLS
                );
            }

            // ---------------------------------------------------------------------------
            // 7) Conv2 backward (grad input -> d_grad_conv1Out)
            // ---------------------------------------------------------------------------
            {
                dim3 block(16, 16, FIRST_OUTPUT_CHANNELS);
                dim3 grid(BATCH_SIZE);

                convBackwardInputKernel<<<grid, block, 0, stream[CurrentBuffer]>>>(
                    d_grad_conv2Out,
                    d_conv2W,
                    d_grad_conv1Out,
                    BATCH_SIZE,
                    FIRST_OUTPUT_CHANNELS,
                    CONV1_OUT_ROWS,
                    CONV1_OUT_COLS,
                    SECOND_OUTPUT_CHANNELS,
                    FILTER_SIZE,
                    FILTER_SIZE,
                    CONV2_OUT_ROWS,
                    CONV2_OUT_COLS
                );
            }

            // ---------------------------------------------------------------------------
            // 8) ReLU backward (Conv1)
            // ---------------------------------------------------------------------------
            {
                int total = BATCH_SIZE * FIRST_OUTPUT_CHANNELS *
                            CONV1_OUT_ROWS * CONV1_OUT_COLS;
                int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

                reluBackwardKernel<<<grid, BLOCK_SIZE, 0, stream[CurrentBuffer]>>>(
                    d_grad_conv1Out,
                    d_conv1Out,
                    d_grad_conv1Out,
                    total
                );
            }

            // ---------------------------------------------------------------------------
            // 9) Conv1 backward (grad W, B)
            // ---------------------------------------------------------------------------
            {
                int totalParams =
                    FIRST_OUTPUT_CHANNELS * FIRST_INPUT_CHANNELS *
                    FILTER_SIZE * FILTER_SIZE +
                    FIRST_OUTPUT_CHANNELS;

                int grid = (totalParams + 31) / 32;

                convBackwardWeightKernel<<<grid, 32, 0, stream[CurrentBuffer]>>>(
                    d_trainImages[CurrentBuffer],
                    d_grad_conv1Out,
                    d_grad_conv1W,
                    d_grad_conv1B,
                    BATCH_SIZE,
                    FIRST_INPUT_CHANNELS,
                    IMAGE_ROWS,
                    IMAGE_COLS,
                    FIRST_OUTPUT_CHANNELS,
                    FILTER_SIZE,
                    FILTER_SIZE,
                    CONV1_OUT_ROWS,
                    CONV1_OUT_COLS
                );
            }

            // ---------------------------------------------------------------------------
            // 10) SGD update (Conv1, Conv2, FC)
            // ---------------------------------------------------------------------------
            {
                auto sgd = [&](float* p, float* g, int n){
                    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    sgdUpdateKernel<<<grid, BLOCK_SIZE, 0, stream[CurrentBuffer]>>>(
                        p, g, LEARNING_RATE, n
                    );
                };

                // Conv1
                sgd(d_conv1W, d_grad_conv1W, conv1W_size);
                sgd(d_conv1B, d_grad_conv1B, FIRST_OUTPUT_CHANNELS);

                // Conv2
                sgd(d_conv2W, d_grad_conv2W, conv2W_size);
                sgd(d_conv2B, d_grad_conv2B, SECOND_OUTPUT_CHANNELS);

                // FC
                sgd(d_fcW, d_grad_fcW, FLATTEN_SIZE * NUM_CLASSES);
                sgd(d_fcB, d_grad_fcB, NUM_CLASSES);
            }

            if(batch + 1 < NUM_BATCHES){
                int nextOffset = (batch + 1) * BATCH_SIZE * (IMAGE_ROWS * IMAGE_COLS);
                memcpy(h_pinnedImages[NextBuffer], &h_trainImages[nextOffset], imageBytesPerBatch);
                memcpy(h_pinnedLabels[NextBuffer], &h_trainLabels[(batch + 1) * BATCH_SIZE], BATCH_SIZE * sizeof(int));
                CudaCheck(cudaMemcpyAsync(d_trainImages[NextBuffer], h_pinnedImages[NextBuffer],
                                           imageBytesPerBatch, cudaMemcpyHostToDevice,
                                           stream[NextBuffer]));
                CudaCheck(cudaMemcpyAsync(d_labels[NextBuffer], h_pinnedLabels[NextBuffer],
                                           BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice,
                                           stream[NextBuffer]));
            }
 
            CudaCheck(cudaStreamSynchronize(stream[CurrentBuffer]));
            float h_loss[BATCH_SIZE];
            CudaCheck(cudaMemcpy(h_loss, d_loss, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            float batchLoss = 0.f;
            for (int i = 0; i < BATCH_SIZE; i++){
                batchLoss += h_loss[i];
            }
            epoch_loss += batchLoss / BATCH_SIZE;
        }
        std::cout << "\n";
        epoch_loss /= NUM_BATCHES;
        printf("Epoch [%d/%d], avg loss=%.6f\n", epoch + 1, EPOCHS, epoch_loss);
    }
 
    for (int i = 0; i < NUM_BUFFERS; i++){
        CudaCheck(cudaStreamSynchronize(stream[i]));
    }
    CudaCheck(cudaEventRecord(stopEvent, 0));
    CudaCheck(cudaEventSynchronize(stopEvent));
    float elapsedTime = 0.f;
    CudaCheck(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

    printf("Time for training %d epochs is: %.2f s\n", EPOCHS, elapsedTime/1000);

    // ---------------------------------------------------------------------------------
    // 6) Testing Phase: Evaluate trained model on MNIST test data (2 Conv architecture)
    // ---------------------------------------------------------------------------------

    cudaEvent_t startEvent_test, stopEvent_test;
    CudaCheck(cudaEventCreate(&startEvent_test));
    CudaCheck(cudaEventCreate(&stopEvent_test));
    CudaCheck(cudaEventRecord(startEvent_test, 0));
    {
        dim3 fcBlock(16, 16);
        dim3 fcGrid((NUM_CLASSES + 15) / 16,
                    (BATCH_SIZE + 15) / 16);
        int fcSharedBytes = 2 * 16 * 16 * sizeof(float);

        int correct = 0;
        int testBatches = TEST_IMAGES / BATCH_SIZE;

        float* h_prob = (float*)malloc(BATCH_SIZE * NUM_CLASSES * sizeof(float));

        for (int batch = 0; batch < testBatches; batch++)
        {
            // -------------------------------------------------
            // Copy input images + labels
            // -------------------------------------------------
            CudaCheck(cudaMemcpy(
                d_trainImages[0],
                &h_testImages[batch * BATCH_SIZE * IMAGE_ROWS * IMAGE_COLS],
                imageBytesPerBatch,
                cudaMemcpyHostToDevice));

            CudaCheck(cudaMemcpy(
                d_labels[0],
                &h_testLabels[batch * BATCH_SIZE],
                BATCH_SIZE * sizeof(int),
                cudaMemcpyHostToDevice));

            // -------------------------------------------------
            // Forward pass
            // -------------------------------------------------

            // ---------- Conv1 + ReLU ----------
            dim3 blockConv1(16, 16);
            dim3 gridConv1(
                (CONV1_OUT_COLS + 15) / 16,
                (CONV1_OUT_ROWS + 15) / 16,
                BATCH_SIZE * FIRST_OUTPUT_CHANNELS);

            convReluKernel<<<gridConv1, blockConv1>>>(
                d_trainImages[0],
                d_conv1W,
                d_conv1B,
                d_conv1Out,
                BATCH_SIZE,
                FIRST_INPUT_CHANNELS,
                IMAGE_ROWS,
                IMAGE_COLS,
                FIRST_OUTPUT_CHANNELS,
                FILTER_SIZE,
                FILTER_SIZE,
                CONV1_OUT_ROWS,
                CONV1_OUT_COLS,
                1,
                0);

            // ---------- Conv2 + ReLU ----------
            dim3 blockConv2(16, 16);
            dim3 gridConv2(
                (CONV2_OUT_COLS + 15) / 16,
                (CONV2_OUT_ROWS + 15) / 16,
                BATCH_SIZE * SECOND_OUTPUT_CHANNELS);

            convReluKernel<<<gridConv2, blockConv2>>>(
                d_conv1Out,
                d_conv2W,
                d_conv2B,
                d_conv2Out,
                BATCH_SIZE,
                SECOND_INPUT_CHANNELS,
                CONV1_OUT_ROWS,
                CONV1_OUT_COLS,
                SECOND_OUTPUT_CHANNELS,
                FILTER_SIZE,
                FILTER_SIZE,
                CONV2_OUT_ROWS,
                CONV2_OUT_COLS,
                1,
                0);

            // ---------- MaxPool + Flatten ----------
            int totalFlat = BATCH_SIZE * SECOND_OUTPUT_CHANNELS *
                            POOL_OUT_ROWS * POOL_OUT_COLS;
            int gridFlat = (totalFlat + BLOCK_SIZE - 1) / BLOCK_SIZE;

            maxPoolFlattenKernel<<<gridFlat, BLOCK_SIZE>>>(
                d_conv2Out,
                d_flat,
                BATCH_SIZE,
                SECOND_OUTPUT_CHANNELS,
                CONV2_OUT_ROWS,
                CONV2_OUT_COLS,
                POOL_SIZE,
                POOL_OUT_ROWS,
                POOL_OUT_COLS);

            // ---------- Fully Connected ----------
            fcForwardKernel<<<fcGrid, fcBlock, fcSharedBytes>>>(
                d_flat,
                d_fcW,
                d_fcB,
                d_fcOut,
                BATCH_SIZE,
                FLATTEN_SIZE,
                NUM_CLASSES);

            // ---------- Softmax ----------
            int gridSoft = (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
            softmaxCrossEntropyKernel<<<gridSoft, BLOCK_SIZE>>>(
                d_fcOut,
                d_labels[0],
                d_loss,
                d_prob,
                BATCH_SIZE,
                NUM_CLASSES);

            CudaCheck(cudaDeviceSynchronize());

            // -------------------------------------------------
            // Accuracy
            // -------------------------------------------------
            CudaCheck(cudaMemcpy(
                h_prob,
                d_prob,
                BATCH_SIZE * NUM_CLASSES * sizeof(float),
                cudaMemcpyDeviceToHost));

            for (int i = 0; i < BATCH_SIZE; i++)
            {
                int pred = 0;
                float best = h_prob[i * NUM_CLASSES];

                for (int c = 1; c < NUM_CLASSES; c++)
                {
                    float p = h_prob[i * NUM_CLASSES + c];
                    if (p > best)
                    {
                        best = p;
                        pred = c;
                    }
                }

                if (pred == h_testLabels[batch * BATCH_SIZE + i])
                    correct++;
            }
        }

        float accuracy = (float)correct / (testBatches * BATCH_SIZE);
        printf("Test accuracy = %.2f%%\n", accuracy * 100.0f);
        CudaCheck(cudaEventRecord(stopEvent_test, 0));
        CudaCheck(cudaEventSynchronize(stopEvent_test));
        float elapsedTime_test = 0.f;
        CudaCheck(cudaEventElapsedTime(&elapsedTime_test, startEvent_test, stopEvent_test));
        printf("Time for testing is: %.2f s\n", elapsedTime_test/1000);

        free(h_prob);
    }
    
    // ---------------------------------------------------------------------------------
    // Cleanup: Free device memory, pinned memory, and streams
    // ---------------------------------------------------------------------------------
    free(h_trainImages);
    free(h_trainLabels);
    free(h_testImages);
    free(h_testLabels);

    for (int i = 0; i < NUM_BUFFERS; i++){
        CudaCheck(cudaFree(d_trainImages[i]));
        CudaCheck(cudaFree(d_labels[i]));
        CudaCheck(cudaFreeHost(h_pinnedImages[i]));
        CudaCheck(cudaFreeHost(h_pinnedLabels[i]));
        CudaCheck(cudaStreamDestroy(stream[i]));
    }


    CudaCheck(cudaFree(d_conv1W));
    CudaCheck(cudaFree(d_conv1B));
    CudaCheck(cudaFree(d_conv1Out));
    CudaCheck(cudaFree(d_conv2W));
    CudaCheck(cudaFree(d_conv2B));
    CudaCheck(cudaFree(d_conv2Out));
    CudaCheck(cudaFree(d_poolOut));
    CudaCheck(cudaFree(d_poolIdx));
    CudaCheck(cudaFree(d_flat));
    CudaCheck(cudaFree(d_fcW));
    CudaCheck(cudaFree(d_fcB));
    CudaCheck(cudaFree(d_fcOut));
    CudaCheck(cudaFree(d_prob));
    CudaCheck(cudaFree(d_loss));
    CudaCheck(cudaFree(d_grad_fcOut));
    CudaCheck(cudaFree(d_grad_fcW));
    CudaCheck(cudaFree(d_grad_fcB));
    CudaCheck(cudaFree(d_grad_flat));
    CudaCheck(cudaFree(d_grad_poolOut));
    CudaCheck(cudaFree(d_grad_conv1Out));
    CudaCheck(cudaFree(d_grad_conv1W));
    CudaCheck(cudaFree(d_grad_conv1B));
    CudaCheck(cudaFree(d_grad_conv2Out));
    CudaCheck(cudaFree(d_grad_conv2W));
    CudaCheck(cudaFree(d_grad_conv2B));
    CudaCheck(cudaFree(d_grad_in));

    return 0;

}
