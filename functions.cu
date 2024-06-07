#include "libs.h"


char *strlwr(char *str)
{
  unsigned char *p = (unsigned char *)str;

  while (*p) {
     *p = tolower((unsigned char)*p);
      p++;
  }

  return str;
}

char get_mode(char* argument) {
    if (argument == NULL) {
        std::cerr << "Error: No argument provided.\n";
        exit(1);
    }
    
    std::string mode_str(strlwr(argument));
    
    
    if (mode_str == "gpu") {
        return 'g';
    } else if (mode_str == "cpu") {
        return 'c';
    } else {
        std::cerr << "Error: Invalid mode. Please use 'gpu' or 'cpu'.\n";
        exit(1);
    }
}


void load_images(std::vector<cv::Mat> *images, int batch_size){
  
    for(int i=0; i<batch_size; i++){
        std::stringstream filename;
        //filename << "images/image1.jpg"; //When loading more than 5k images
        filename << "images/image" << i+1 << ".jpg";
        //load images
        cv::Mat image = cv::imread(filename.str(), CV_LOAD_IMAGE_COLOR);
        image.convertTo(image, CV_32FC3);
        cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);

        images->push_back(image);        
    }
}


void save_images(float *h_output_tensor, int rows, int cols, int batch_size){
    //Cleaning the directory before populating it
    system("rm -r outputs/gpu/*");
    //Saving output to image
    for(int i=0; i<batch_size; i++){
        int offset = i * 3 * rows * cols;
        cv::Mat output_image(rows, cols, CV_32FC3, h_output_tensor + offset);
        //Tresholding to remove negative values
        cv::threshold(output_image,
                      output_image,
                      0, //threshold
                      0, //maxval
                      cv::THRESH_TOZERO);
        cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
        output_image.convertTo(output_image, CV_8UC1);
        std::stringstream filename;
        filename << "outputs/gpu/cudnn_output_" << i+1 << ".jpg";
        cv::imwrite(filename.str(), output_image);
    }
}
//Find a way to use just one of them
//Also, delete the content of the respective folder
void save_images_cpu(std::vector<cv::Mat> images, int batch_size){
    //Cleaning the directory before populating it
    system("rm -r outputs/cpu/*");
    //Saving output to image
    for(int i=0; i<batch_size; i++){
        cv::Mat image = images.at(i).clone();
        //Tresholding to remove negative values
        cv::threshold(image,
                      image,
                      0, //threshold
                      0, //maxval
                      cv::THRESH_TOZERO);
        cv::normalize(image, image, 0.0, 255.0, cv::NORM_MINMAX);
        //image.convertTo(image, CV_8UC1);
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        std::stringstream filename;
        filename << "outputs/cpu/cpu_output_" << i+1 << ".jpg";
        cv::imwrite(filename.str(), image);
    }
}