#include "config.h"
#include "utils/load_data.h"
#include "kernels/forward.h"
#include "kernels/backward.h"
#include <cuda_runtime.h>
#include <iostream>

inline void CudaCheck(cudaError_t err, const char* msg = "") {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s (%s)\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Print the progress bar during the training
 * 
 * @param current current batch idx
 * @param total total number of batches
 */
void print_progress(int current, int total) {
    const int bar_width = 40;
    float progress = (float)current / total;
    int pos = bar_width * progress;

    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "%";
    std::cout.flush();
}

/**
 * @brief the code implements the training of a CNN network in CUDA using SGD
 * 
 * @param argc 
 * @param argv use the `mnist` (defualt) or the `fashion` flag for the dataset  
 */
int main(int argc, char* argv[]) {

    float* h_train_images = (float*) malloc(TRAIN_IMAGES * IMAGE_ROWS * IMAGE_COLS * sizeof(float)); 
    float* h_test_images = (float*) malloc(TEST_IMAGES * IMAGE_ROWS * IMAGE_COLS * sizeof(float));
    int* h_train_labels = (int*) malloc(TRAIN_IMAGES * sizeof(int));
    int* h_test_labels = (int*) malloc(TEST_IMAGES * sizeof(int));

    if (!h_train_images || !h_test_images || !h_train_labels || !h_test_labels) {
        printf("Memory allocation failed\n");
        return 1;
    }

    /* 
        Load the dataset
    */
    std::string dataset = "mnist"; 

    if (argc == 2) {
        dataset = argv[1];
    }

    std::string train_images = "../datasets/" + dataset + "/train-images.idx3-ubyte";
    std::string test_images ="../datasets/" + dataset + "/t10k-images.idx3-ubyte";
    std::string train_labels = "../datasets/" + dataset + "/train-labels.idx1-ubyte";
    std::string test_labels = "../datasets/" + dataset + "/t10k-labels.idx1-ubyte";

    load_image(train_images.c_str(), h_train_images, TRAIN_IMAGES);
    load_image(test_images.c_str(), h_test_images, TEST_IMAGES);
    load_labels(train_labels.c_str(), h_train_labels, TRAIN_IMAGES);
    load_labels(test_labels.c_str(), h_test_labels, TEST_IMAGES);

    printf("Network --> (CONV(1, %d, %d), RELU) + (CONV(%d, %d, %d), RELU) + MaxPool(%d) + Flatten + FC + SoftMax\n", FIRST_OUTPUT_CHANNELS, FILTER_SIZE, FIRST_OUTPUT_CHANNELS, SECOND_OUTPUT_CHANNELS, FILTER_SIZE, POOL_SIZE);
    printf("Epochs: %d, Learning rate: %.2f, Batch size: %d\n", EPOCHS, LEARNING_RATE, BATCH_SIZE);
    printf("Block Size: %d\n", BLOCK_SIZE);


    // --------------------
    // Pointers
    // --------------------
    float* d_train_images;
    int*   d_labels;

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

    // Dimensions
    int conv1W_size = FIRST_OUTPUT_CHANNELS * FIRST_INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE;
    int conv2W_size = SECOND_OUTPUT_CHANNELS * SECOND_INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE;
    size_t imageBytes = BATCH_SIZE * IMAGE_ROWS * IMAGE_COLS * sizeof(float);

    // --------------------
    // Dynamic memory allocation
    // --------------------
    CudaCheck(cudaMalloc(&d_train_images, BATCH_SIZE * IMAGE_ROWS * IMAGE_COLS * sizeof(float)));
    CudaCheck(cudaMalloc(&d_labels, BATCH_SIZE * sizeof(int)));

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

    CudaCheck(cudaMalloc(&d_grad_fcW, FLATTEN_SIZE*NUM_CLASSES*sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_fcB, NUM_CLASSES*sizeof(float)));

    CudaCheck(cudaMalloc(&d_grad_conv1Out, BATCH_SIZE * FIRST_OUTPUT_CHANNELS * CONV1_OUT_ROWS * CONV1_OUT_COLS * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_conv1W, conv1W_size * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_conv1B, FIRST_OUTPUT_CHANNELS * sizeof(float)));

    CudaCheck(cudaMalloc(&d_grad_conv2Out, BATCH_SIZE * SECOND_OUTPUT_CHANNELS * CONV2_OUT_ROWS * CONV2_OUT_COLS * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_conv2W, conv2W_size * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_conv2B, SECOND_OUTPUT_CHANNELS * sizeof(float)));

    CudaCheck(cudaMalloc(&d_grad_poolOut, BATCH_SIZE * SECOND_OUTPUT_CHANNELS * POOL_OUT_ROWS * POOL_OUT_COLS * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_in, BATCH_SIZE * IMAGE_ROWS * IMAGE_COLS * sizeof(float)));

    // --------------------
    //  He weights initialization on host and copied to device
    // --------------------

    /*
        He initialization, is better suited for layers that use ReLU activation functions since it 
        mitigates the exploding gradient issue.
        The layer weights are initialized in the range [-limit, +limit] while the bias are initialized to 0.
        The limit is W ~ U(- sqrt(6/n), sqrt(6/n)) where n is the number of input neurons to the layer.
    */
    srand(21);

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
    free(h_fcW);
    free(h_fcB);

    // Record time for training and testing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    printf("Starting training for %d epochs:\n", EPOCHS);

    for (int epoch=0; epoch < EPOCHS; epoch++){
        float epoch_loss = 0.0;

        for (int batch=0; batch < NUM_BATCHES; batch++){
            
            print_progress(batch + 1, NUM_BATCHES);

            // Copy on device the batch images and labels
            CudaCheck(cudaMemcpy(d_train_images, &h_train_images[(batch *BATCH_SIZE) * (IMAGE_ROWS * IMAGE_COLS)], imageBytes, cudaMemcpyHostToDevice));
            CudaCheck(cudaMemcpy(d_labels, &h_train_labels[batch * BATCH_SIZE], sizeof(int) * BATCH_SIZE, cudaMemcpyHostToDevice));

            /*-----------------
                FORWARD PASS
            -----------------*/

            int total, grid;

            // Conv1 -> ReLU
            total = BATCH_SIZE * FIRST_OUTPUT_CHANNELS * CONV1_OUT_ROWS * CONV1_OUT_COLS;
            grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            ConvForward<<<grid, BLOCK_SIZE>>>(d_train_images, d_conv1W, d_conv1B, d_conv1Out,
                                            BATCH_SIZE, FIRST_INPUT_CHANNELS, FIRST_OUTPUT_CHANNELS,
                                            IMAGE_ROWS, IMAGE_COLS, FILTER_SIZE);

            ReLUForward<<<grid, BLOCK_SIZE>>>(d_conv1Out, total);

            // Conv2 -> ReLU
            total = BATCH_SIZE * SECOND_OUTPUT_CHANNELS * CONV2_OUT_ROWS * CONV2_OUT_COLS;
            grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            ConvForward<<<grid, BLOCK_SIZE>>>(d_conv1Out, d_conv2W, d_conv2B, d_conv2Out,
                                            BATCH_SIZE, SECOND_INPUT_CHANNELS, SECOND_OUTPUT_CHANNELS,
                                            CONV1_OUT_ROWS, CONV1_OUT_COLS, FILTER_SIZE);

            ReLUForward<<<grid, BLOCK_SIZE>>>(d_conv2Out, total);

            // MaxPool
            total = BATCH_SIZE * SECOND_OUTPUT_CHANNELS * POOL_OUT_ROWS * POOL_OUT_COLS;
            grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            MaxPoolForward<<<grid, BLOCK_SIZE>>>(d_conv2Out, d_poolOut, d_poolIdx, BATCH_SIZE, SECOND_OUTPUT_CHANNELS, CONV2_OUT_ROWS, CONV2_OUT_COLS, POOL_SIZE);

            CudaCheck(cudaMemset(d_flat, 0, BATCH_SIZE*FLATTEN_SIZE*sizeof(float)));
            total = BATCH_SIZE*SECOND_OUTPUT_CHANNELS*POOL_OUT_ROWS*POOL_OUT_COLS;
            grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
            FlattenForward<<<grid,BLOCK_SIZE>>>(d_poolOut, d_flat, BATCH_SIZE, SECOND_OUTPUT_CHANNELS, POOL_OUT_ROWS, POOL_OUT_COLS);

            total = BATCH_SIZE*NUM_CLASSES;
            grid = (total + BLOCK_SIZE - 1)/BLOCK_SIZE;
            FullyConnectedForward<<<grid,BLOCK_SIZE>>>(d_flat, d_fcW, d_fcB, d_fcOut, BATCH_SIZE, NUM_CLASSES, FLATTEN_SIZE);

            grid = (BATCH_SIZE + BLOCK_SIZE - 1)/BLOCK_SIZE;
            SoftmaxCrossEntropyForward<<<grid, BLOCK_SIZE>>>(d_fcOut, d_labels, d_loss, d_prob, BATCH_SIZE, NUM_CLASSES);
            
            // Compute batch loss on host
            float h_loss[BATCH_SIZE];
            CudaCheck(cudaMemcpy(h_loss, d_loss, BATCH_SIZE*sizeof(float), cudaMemcpyDeviceToHost));
            float batchLoss = 0.0f;
            for(int i=0; i<BATCH_SIZE; i++)
                batchLoss += h_loss[i];
            epoch_loss += batchLoss / BATCH_SIZE;


            // ---------------------------
            // BACKWARD PASS
            // ---------------------------

            // Softmax + CrossEntropy
            total = BATCH_SIZE;
            grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            SoftmaxCrossEntropyBackward<<<grid, BLOCK_SIZE>>>(d_grad_fcOut, d_prob, d_labels, BATCH_SIZE, NUM_CLASSES);

            // Fully Connected Layer gradients
            CudaCheck(cudaMemset(d_grad_fcW, 0, FLATTEN_SIZE * NUM_CLASSES * sizeof(float)));
            CudaCheck(cudaMemset(d_grad_fcB, 0, NUM_CLASSES * sizeof(float)));
            int totalParams = FLATTEN_SIZE * NUM_CLASSES + NUM_CLASSES;
            grid = (totalParams + BLOCK_SIZE - 1) / BLOCK_SIZE;
            FullyConnectedLayerBackward<<<grid, BLOCK_SIZE>>>(d_grad_fcOut, d_flat, d_grad_fcW, d_grad_fcB,
                                                            BATCH_SIZE, NUM_CLASSES, FLATTEN_SIZE);

            // Gradient w.r.t. flattened input
            total = BATCH_SIZE * FLATTEN_SIZE;
            grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            FullyConnectedBackward<<<grid, BLOCK_SIZE>>>(d_grad_fcOut, d_fcW, d_grad_flat,
                                                        BATCH_SIZE, NUM_CLASSES, FLATTEN_SIZE);

            // Unflatten to pooled shape
            total = BATCH_SIZE * SECOND_OUTPUT_CHANNELS * POOL_OUT_ROWS * POOL_OUT_COLS;
            grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            FlattenBackward<<<grid, BLOCK_SIZE>>>(d_grad_flat, d_grad_poolOut, BATCH_SIZE,
                                                SECOND_OUTPUT_CHANNELS, POOL_OUT_ROWS, POOL_OUT_COLS, FLATTEN_SIZE);

            // MaxPool backward
            cudaMemset(d_grad_conv2Out, 0, BATCH_SIZE * SECOND_OUTPUT_CHANNELS * CONV2_OUT_ROWS * CONV2_OUT_COLS * sizeof(float));
            total = BATCH_SIZE * SECOND_OUTPUT_CHANNELS * POOL_OUT_ROWS * POOL_OUT_COLS;
            grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            MaxPoolBackward<<<grid, BLOCK_SIZE>>>(d_grad_poolOut, d_grad_conv2Out, d_poolIdx,
                                                BATCH_SIZE, SECOND_OUTPUT_CHANNELS, POOL_OUT_ROWS, POOL_OUT_COLS);

            // ReLU backward for conv2
            total = BATCH_SIZE * SECOND_OUTPUT_CHANNELS * CONV2_OUT_ROWS * CONV2_OUT_COLS;
            grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            ReLUBackward<<<grid, BLOCK_SIZE>>>(d_grad_conv2Out, d_conv2Out, d_grad_conv2Out, total);

           // Conv2 gradients (weights & bias)
            cudaMemset(d_grad_conv2W, 0, conv2W_size * sizeof(float));
            cudaMemset(d_grad_conv2B, 0, SECOND_OUTPUT_CHANNELS * sizeof(float));
            totalParams = SECOND_OUTPUT_CHANNELS * FIRST_OUTPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE + SECOND_OUTPUT_CHANNELS;
            grid = (totalParams + BLOCK_SIZE - 1) / BLOCK_SIZE;
            ConvLayerBackward<<<grid, BLOCK_SIZE>>>(
                d_conv1Out,          // input to conv2
                d_grad_conv2Out,     // grad w.r.t conv2 output
                d_grad_conv2W,       // grad weights
                d_grad_conv2B,       // grad biases
                BATCH_SIZE,          // batch size
                FIRST_OUTPUT_CHANNELS, // input channels
                SECOND_OUTPUT_CHANNELS, // output channels
                CONV1_OUT_ROWS,      // input rows
                CONV1_OUT_COLS,      // input cols
                CONV2_OUT_ROWS,      // output rows
                CONV2_OUT_COLS,      // output cols
                FILTER_SIZE          // filter size
            );

            // Gradient w.r.t. conv1 output (input of conv2)
            total = BATCH_SIZE * FIRST_OUTPUT_CHANNELS * CONV1_OUT_ROWS * CONV1_OUT_COLS;
            grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            ConvBackward<<<grid, BLOCK_SIZE>>>(
                d_grad_conv2Out,     // grad w.r.t conv2 output
                d_conv2W,            // conv2 weights
                d_grad_conv1Out,     // grad w.r.t conv1 output
                BATCH_SIZE,
                FIRST_OUTPUT_CHANNELS,  // input channels
                SECOND_OUTPUT_CHANNELS, // output channels
                CONV1_OUT_ROWS,         // input rows
                CONV1_OUT_COLS,         // input cols
                CONV2_OUT_ROWS,         // output rows
                CONV2_OUT_COLS,         // output cols
                FILTER_SIZE
            );

            // ReLU backward for conv1
            total = BATCH_SIZE * FIRST_OUTPUT_CHANNELS * CONV1_OUT_ROWS * CONV1_OUT_COLS;
            grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            ReLUBackward<<<grid, BLOCK_SIZE>>>(
                d_grad_conv1Out,
                d_conv1Out,
                d_grad_conv1Out,
                total
            );

            // Conv1 gradients (weights & bias)
            cudaMemset(d_grad_conv1W, 0, conv1W_size * sizeof(float));
            cudaMemset(d_grad_conv1B, 0, FIRST_OUTPUT_CHANNELS * sizeof(float));
            totalParams = FIRST_OUTPUT_CHANNELS * FIRST_INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE + FIRST_OUTPUT_CHANNELS;
            grid = (totalParams + BLOCK_SIZE - 1) / BLOCK_SIZE;
            ConvLayerBackward<<<grid, BLOCK_SIZE>>>(
                d_train_images,      // input images
                d_grad_conv1Out,     // grad w.r.t conv1 output
                d_grad_conv1W,       // grad weights
                d_grad_conv1B,       // grad biases
                BATCH_SIZE,
                FIRST_INPUT_CHANNELS,
                FIRST_OUTPUT_CHANNELS,
                IMAGE_ROWS,
                IMAGE_COLS,
                CONV1_OUT_ROWS,
                CONV1_OUT_COLS,
                FILTER_SIZE
            );

            //  Gradient w.r.t. input images (optional)
            total = BATCH_SIZE * FIRST_INPUT_CHANNELS * IMAGE_ROWS * IMAGE_COLS;
            grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            ConvBackward<<<grid, BLOCK_SIZE>>>(
                d_grad_conv1Out,     // grad w.r.t conv1 output
                d_conv1W,            // conv1 weights
                d_grad_in,           // grad w.r.t input images
                BATCH_SIZE,
                FIRST_INPUT_CHANNELS,
                FIRST_OUTPUT_CHANNELS,
                IMAGE_ROWS,
                IMAGE_COLS,
                CONV1_OUT_ROWS,
                CONV1_OUT_COLS,
                FILTER_SIZE
            );


            // ---------------------------
            // UPDATE PARAMETERS
            // ---------------------------

            // Conv1 parameters
            total = conv1W_size;
            grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            SGDBackward<<<grid, BLOCK_SIZE>>>(d_conv1W, d_grad_conv1W, LEARNING_RATE, total);

            total = FIRST_OUTPUT_CHANNELS;
            grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            SGDBackward<<<grid, BLOCK_SIZE>>>(d_conv1B, d_grad_conv1B, LEARNING_RATE, total);

            // Conv2 parameters
            total = conv2W_size;
            grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            SGDBackward<<<grid, BLOCK_SIZE>>>(d_conv2W, d_grad_conv2W, LEARNING_RATE, total);

            total = SECOND_OUTPUT_CHANNELS;
            grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            SGDBackward<<<grid, BLOCK_SIZE>>>(d_conv2B, d_grad_conv2B, LEARNING_RATE, total);

            // Fully Connected Layer parameters
            total = FLATTEN_SIZE * NUM_CLASSES;
            grid      = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            SGDBackward<<<grid, BLOCK_SIZE>>>(d_fcW, d_grad_fcW, LEARNING_RATE, total);

            total = NUM_CLASSES;
            grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            SGDBackward<<<grid, BLOCK_SIZE>>>(d_fcB, d_grad_fcB, LEARNING_RATE, total);


        }
        std::cout << "\n";


        // ---------------------------
        // Batch loss logging
        // ---------------------------
        epoch_loss /= NUM_BATCHES;
        printf("Epoch [%d/%d], avg loss = %.6f\n", epoch+1, EPOCHS, epoch_loss);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);  // Wait for the stop event to complete
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time for training %d epochs is: %.2f s\n", EPOCHS, elapsedTime/1000);

    // ---------------------------------------------------------------------------
    // 5) Evaluate on test set
    // ---------------------------------------------------------------------------
    int correct = 0;
    int testBatches = TEST_IMAGES / BATCH_SIZE;
    cudaEvent_t start_test, stop_test;
    cudaEventCreate(&start_test);
    cudaEventCreate(&stop_test);
    cudaEventRecord(start_test, 0);

    for(int b = 0; b < testBatches; b++) {
        // Copia batch di immagini e label sul device
        CudaCheck(cudaMemcpy(d_train_images,
                &h_test_images[b * BATCH_SIZE * IMAGE_ROWS * IMAGE_COLS],
                BATCH_SIZE * IMAGE_ROWS * IMAGE_COLS * sizeof(float), cudaMemcpyHostToDevice));
        CudaCheck(cudaMemcpy(d_labels,
                &h_test_labels[b * BATCH_SIZE],
                BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));

        int total, grid;

        // Conv1 -> ReLU
        total = BATCH_SIZE * FIRST_OUTPUT_CHANNELS * CONV1_OUT_ROWS * CONV1_OUT_COLS;
        grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        ConvForward<<<grid, BLOCK_SIZE>>>(d_train_images, d_conv1W, d_conv1B, d_conv1Out,
                                        BATCH_SIZE, FIRST_INPUT_CHANNELS, FIRST_OUTPUT_CHANNELS,
                                        IMAGE_ROWS, IMAGE_COLS, FILTER_SIZE);

        ReLUForward<<<grid, BLOCK_SIZE>>>(d_conv1Out, total);


        // Conv2 -> ReLU
        total = BATCH_SIZE * SECOND_OUTPUT_CHANNELS * CONV2_OUT_ROWS * CONV2_OUT_COLS;
        grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        ConvForward<<<grid, BLOCK_SIZE>>>(d_conv1Out, d_conv2W, d_conv2B, d_conv2Out,
                                        BATCH_SIZE, SECOND_INPUT_CHANNELS, SECOND_OUTPUT_CHANNELS,
                                        CONV1_OUT_ROWS, CONV1_OUT_COLS, FILTER_SIZE);

        ReLUForward<<<grid, BLOCK_SIZE>>>(d_conv2Out, total);


        // MaxPool
        total = BATCH_SIZE * SECOND_OUTPUT_CHANNELS * POOL_OUT_ROWS * POOL_OUT_COLS;
        grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        MaxPoolForward<<<grid, BLOCK_SIZE>>>(d_conv2Out, d_poolOut, d_poolIdx,
                                            BATCH_SIZE, SECOND_OUTPUT_CHANNELS,
                                            CONV2_OUT_ROWS, CONV2_OUT_COLS, POOL_SIZE);


        // Flatten
        FlattenForward<<<grid, BLOCK_SIZE>>>(d_poolOut, d_flat,
                                            BATCH_SIZE, SECOND_OUTPUT_CHANNELS,
                                            POOL_OUT_ROWS, POOL_OUT_COLS);


        // Fully connected
        total = BATCH_SIZE * NUM_CLASSES;
        grid  = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        FullyConnectedForward<<<grid, BLOCK_SIZE>>>(d_flat, d_fcW, d_fcB, d_fcOut,
                                                    BATCH_SIZE, NUM_CLASSES, FLATTEN_SIZE);


        // Softmax (solo per predizioni)
        grid = (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
        SoftmaxCrossEntropyForward<<<grid, BLOCK_SIZE>>>(d_fcOut, d_labels, d_loss, d_prob,
                                                        BATCH_SIZE, NUM_CLASSES);


        // Copy prediction on host
        float* h_prob = (float*)malloc(BATCH_SIZE * NUM_CLASSES * sizeof(float));
        CudaCheck(cudaMemcpy(h_prob, d_prob, BATCH_SIZE * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost));

        int* h_lbl = (int*)malloc(BATCH_SIZE * sizeof(int));
        memcpy(h_lbl, &h_test_labels[b * BATCH_SIZE], BATCH_SIZE * sizeof(int));

        // Count correct predictions
        for(int i = 0; i < BATCH_SIZE; i++) {
            int pred = 0;
            float maxp = h_prob[i * NUM_CLASSES];
            for(int c = 1; c < NUM_CLASSES; c++) {
                if(h_prob[i * NUM_CLASSES + c] > maxp) {
                    maxp = h_prob[i * NUM_CLASSES + c];
                    pred = c;
                }
            }
            if(pred == h_lbl[i]) correct++;
        }

        free(h_prob);
        free(h_lbl);
    }

    // Compute accuracy
    float accuracy = static_cast<float>(correct) / (testBatches * BATCH_SIZE);
    printf("Test accuracy = %.2f%%\n", accuracy * 100.f);

    // Compute total time
    cudaEventRecord(stop_test, 0);
    cudaEventSynchronize(stop_test);
    float elapsedTime_test;
    cudaEventElapsedTime(&elapsedTime_test, start_test, stop_test);
    printf("Time for testing is: %.2f s\n", elapsedTime_test / 1000.f);


    // Free memory
    free(h_train_images);
    free(h_test_images);
    free(h_train_labels);
    free(h_test_labels);

    CudaCheck(cudaFree(d_train_images));
    CudaCheck(cudaFree(d_labels));
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