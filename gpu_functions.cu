#include "libs.h"

void gpu_mode(int batch_size)
{
    cudnnHandle_t cudnn_handle;
    cudaEvent_t start, stop; // CUDA TIMING
    float time;
    std::vector<cv::Mat> images;
    int channels = 3;

    // Creating timing events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Loading images
    std::cout << "Loading images...\n";
    load_images(&images, batch_size);
    int image_rows = images.at(0).rows;
    int image_cols = images.at(0).cols;
    checkCUDNN(cudnnCreate(&cudnn_handle));
    std::cout << "Preparing descriptors...\n";
    // Initializing descriptors
    cudnnTensorDescriptor_t input_tensor;
    cudnnTensorDescriptor_t output_tensor;
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    initializing_descriptors(&input_tensor, &output_tensor, &kernel_descriptor,
                             &convolution_descriptor, batch_size, channels, image_rows, image_cols);

    // Defining the algorithm
    cudnnConvolutionFwdAlgo_t convolution_algorithm = defining_algorithm(cudnn_handle, input_tensor, output_tensor, kernel_descriptor, convolution_descriptor);

    // Allocating memory
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                       input_tensor,
                                                       kernel_descriptor,
                                                       convolution_descriptor,
                                                       output_tensor,
                                                       convolution_algorithm,
                                                       &workspace_bytes));

    void *d_workspace;     // Device workspace memory
    void *d_input_tensor;  // Device input tensor memory
    void *d_output_tensor; // Device output tensor memory
    void *d_kernel;        // Device kernel memory
    int tensor_size = batch_size * channels * image_rows * image_cols * sizeof(float);
    std::cout << "Allocating device memory...\n";

    // Start timer.
    std::cout << "Timer starts now.\n";
    cudaEventRecord(start, 0);

    cudaMalloc(&d_workspace, workspace_bytes);
    checkCUDAError("cudaMalloc");
    allocate_memory(images, &d_input_tensor, &d_output_tensor, &d_kernel, tensor_size);

    // Performing algorithm
    std::cout << "Performing the convolution...\n";
    const float alpha = 1, beta = 0;
    cudnnConvolutionForward(cudnn_handle,
                            &alpha,
                            input_tensor,
                            d_input_tensor,
                            kernel_descriptor,
                            d_kernel,
                            convolution_descriptor,
                            convolution_algorithm,
                            d_workspace,
                            workspace_bytes,
                            &beta,
                            output_tensor,
                            d_output_tensor);

    // Moving result back to CPU
    float *h_output_tensor;
    h_output_tensor = (float *)malloc(batch_size * channels * image_rows * image_cols * sizeof(float));
    cudaMemcpy(h_output_tensor, d_output_tensor, tensor_size, cudaMemcpyDeviceToHost);

    // Stop timer.
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Estimating time difference between start and stop, then storing the result in time.
    cudaEventElapsedTime(&time, start, stop);

    // Saving images
    save_images(h_output_tensor, image_rows, image_cols, batch_size);

    printf("Execution time for convolution: %8.2f milliseconds\n", time);
    std::cout << "Results can be found in the outputs/gpu/ folder.\n";

    // Cleaning memory
    cudnnDestroyTensorDescriptor(input_tensor);
    cudnnDestroyTensorDescriptor(output_tensor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudaFree(d_workspace);
    cudaFree(d_input_tensor);
    cudaFree(d_output_tensor);
    cudaFree(d_kernel);
    free(h_output_tensor);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudnnDestroy(cudnn_handle);
}

void initializing_descriptors(cudnnTensorDescriptor_t *input_tensor,
                              cudnnTensorDescriptor_t *output_tensor,
                              cudnnFilterDescriptor_t *kernel_descriptor,
                              cudnnConvolutionDescriptor_t *convolution_descriptor,
                              int batch_size,
                              int channels,
                              int image_rows,
                              int image_cols)
{

    // Input tensor descriptor
    checkCUDNN(cudnnCreateTensorDescriptor(input_tensor));
    checkCUDNN(cudnnSetTensor4dDescriptor(*input_tensor,
                                          CUDNN_TENSOR_NHWC, // format
                                          CUDNN_DATA_FLOAT,  // data type
                                          batch_size,        // batch size
                                          channels,          // channels
                                          image_rows,        // height
                                          image_cols));      // width
    // Output tensor descriptor
    checkCUDNN(cudnnCreateTensorDescriptor(output_tensor));
    checkCUDNN(cudnnSetTensor4dDescriptor(*output_tensor,
                                          CUDNN_TENSOR_NHWC, // format
                                          CUDNN_DATA_FLOAT,  // data type
                                          batch_size,        // batch size
                                          channels,          // channels
                                          image_rows,        // height
                                          image_cols));      // width

    // Kernel descriptor
    checkCUDNN(cudnnCreateFilterDescriptor(kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(*kernel_descriptor,
                                          CUDNN_DATA_FLOAT,  // data type
                                          CUDNN_TENSOR_NCHW, // format
                                          channels,          // out channels
                                          channels,          // in channels
                                          3,                 // height
                                          3));               // width

    // Convolution descriptor
    checkCUDNN(cudnnCreateConvolutionDescriptor(convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(*convolution_descriptor,
                                               1,                       // pad height
                                               1,                       // pad width
                                               1,                       // vertical_stride
                                               1,                       // horizontal_stride
                                               1,                       // dilation_height
                                               1,                       // dilation_width
                                               CUDNN_CROSS_CORRELATION, // mode
                                               CUDNN_DATA_FLOAT));      // computeType
}

cudnnConvolutionFwdAlgo_t defining_algorithm(cudnnHandle_t cudnn_handle,
                                             cudnnTensorDescriptor_t input_tensor,
                                             cudnnTensorDescriptor_t output_tensor,
                                             cudnnFilterDescriptor_t kernel_descriptor,
                                             cudnnConvolutionDescriptor_t convolution_descriptor){
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    // Convolution algorithm
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn_handle,
                                                   input_tensor,
                                                   kernel_descriptor,
                                                   convolution_descriptor,
                                                   output_tensor,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                   0, // Memory limit in bytes
                                                   &convolution_algorithm));
    return convolution_algorithm;
}

void allocate_memory(std::vector<cv::Mat> images, void **d_input_tensor,
                        void **d_output_tensor, void **d_kernel, int tensor_size)
{
    // Allocating device input tensor memory
    cudaMalloc(d_input_tensor, tensor_size);
    checkCUDAError("cudaMalloc");
    int rows = images.at(0).rows;
    int cols = images.at(0).cols;
    for (int i = 0; i < images.size(); i++)
    {
        size_t offset = i * 3 * rows * cols * sizeof(float);
        cudaMemcpy(static_cast<char *>(*d_input_tensor) + offset,
                   images[i].ptr<float>(0), 3 * rows * cols * sizeof(float),
                   cudaMemcpyHostToDevice);
        checkCUDAError("cudaMemcpy");
    }
    // Allocating device output tensor memory
    cudaMalloc(d_output_tensor, tensor_size);
    checkCUDAError("cudaMalloc");
    cudaMemset(*d_output_tensor, 0, tensor_size);
    checkCUDAError("cudaMemset");
    // Allocating device kernel memory
    float h_kernel[3][3][3][3];
    for (int kernel = 0; kernel < 3; ++kernel){
        for (int channel = 0; channel < 3; ++channel){
            for (int row = 0; row < 3; ++row){
                for (int column = 0; column < 3; ++column){
                    h_kernel[kernel][channel][row][column] = kernel_template[row][column];
                }
            }
        }
    }
    cudaMalloc(d_kernel, sizeof(h_kernel));
    checkCUDAError("cudaMalloc");
    cudaMemcpy(*d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy");
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // cudaGetErrorString(err) prende l'errore in input, che e' di tipo cudastranoerror, e lo trasforma in una stringa stampabile a schermo.
        fprintf(stderr, "Cuda error: %s %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void checkCUDNN(cudnnStatus_t expression)
{
    cudnnStatus_t status = (expression);
    if (status != CUDNN_STATUS_SUCCESS)
    {
        std::cerr << "Error on line " << __LINE__ << ": "
                  << cudnnGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}