#ifndef MAIN_H
#define MAIN_H

#include <stdio.h>
#include <cuda.h>
#include <cudnn.h>
#include <string>
#include <sstream>
#include <time.h>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

#define GRN   "\x1B[32m"
#define BLU   "\x1B[34m"
#define RESET "\x1B[0m"

//CPU related functions
void cpu_mode(int );
//GPU related functions
void gpu_mode(int );
void initializing_descriptors(cudnnTensorDescriptor_t *, cudnnTensorDescriptor_t *, cudnnFilterDescriptor_t *, cudnnConvolutionDescriptor_t *, int, int, int, int);
cudnnConvolutionFwdAlgo_t defining_algorithm(cudnnHandle_t, cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnFilterDescriptor_t, cudnnConvolutionDescriptor_t);
void allocate_memory(std::vector<cv::Mat>, void **, void **, void **, int);
void checkCUDAError(const char*);
void checkCUDNN(cudnnStatus_t );
//misc functions
char *strlwr(char *);
char get_mode(char *);
void load_images(std::vector<cv::Mat> *, int);
void save_images(float *, int, int, int);
void save_images_cpu(std::vector<cv::Mat>, int);
const float kernel_template[3][3] = { //Laplacian filter
  {0,  1, 0},
  {1, -4, 1},
  {0,  1, 0}
};

#endif /* MAIN_H */