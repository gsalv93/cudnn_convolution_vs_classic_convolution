#include "libs.h"


void cpu_mode(int batch_size){
    cudaEvent_t start, stop; //CUDA TIMING
    float time;
    std::vector<cv::Mat> images;
    cv::Mat kernel(3, 3, CV_32FC1, (void *)kernel_template);
    //Creating timing events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //Loading images
    std::cout<<"Loading images...\n";
    load_images(&images, batch_size);

    std::vector<cv::Mat> output_images;

    std::cout<<"Performing the convolution...\n";
    //Start timer
    std::cout<<"Timer starts now.\n";
    cudaEventRecord(start, 0);
    for(int i=0; i<batch_size; i++){
        cv::Mat convolved_image = cv::Mat::zeros(images.at(i).rows, images.at(i).cols, CV_32FC3);
        cv::filter2D(images.at(i), convolved_image, images.at(i).depth(), kernel);
        output_images.push_back(convolved_image);
    }

    //Stop timer.
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    //Estimating time difference between start and stop, then storing the result in time.
    cudaEventElapsedTime(&time, start, stop);

    printf("Execution time for convolution: %8.2f milliseconds\n", time);
    std::cout<<"Results can be found in the outputs/cpu/ folder.\n";


    //Saving results to images
    save_images_cpu(output_images, batch_size);
    //Cleaning
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

