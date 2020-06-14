#include <stdint.h>

#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <opencv2/opencv.hpp>

#include "vo.h"
#include "params.h"
#include "feature.h"


const bool kVisualize = false;

const int kImageHeight = 192;
const int kImageWidth  = 640;


int main (int argc, char** argv) {
  if (argc<2) {
    std::cerr << "Usage: ./HomeworkVO path/to/sequence/00" << std::endl;
    return 1;
  }

  std::string sequence_directory = argv[1];

  // set camera parameters
  Params params;
  params.K.fx = 370.75;
  params.K.fy = 367.08;
  params.K.cu = 313.15;
  params.K.cv = 94.58;
  params.baseline = 0.54;

  // pose from camera to world
  Eigen::Matrix<double,4,4> pose = Eigen::MatrixXd::Identity(4,4);

  for (int i = 0; i < 4541; i++) {
    // image file name
    char img_name[256]; sprintf(img_name, "%06d.png", i);
    std::string left_img_name  = sequence_directory + "/image_0/" + img_name;
    std::string right_img_name = sequence_directory + "/image_1/" + img_name;

    cv::Mat left_img  = cv::imread(left_img_name,  CV_8UC1);
    cv::Mat right_img = cv::imread(right_img_name, CV_8UC1);

    if (left_img.empty() || right_img.empty()) {
      std::cerr << "Read images failed!" << std::endl;
      return 2;
    }

    cv::resize(left_img,  left_img,  cv::Size(640, 192), cv::INTER_LINEAR);
    cv::resize(right_img, right_img, cv::Size(640, 192), cv::INTER_LINEAR);

    // entry to algorithm
    ProcessFrame(left_img, right_img);



  }
  // // init visual odometry
  // VisualOdometryStereo viso(param);
  //   
  // // loop through all frames i=0:372
  // for (int32_t i=0; i<373; i++) {

  //   // input file names
  //   char base_name[256]; sprintf(base_name,"%06d.png",i);
  //   string left_img_file_name  = dir + "/I1_" + base_name;
  //   string right_img_file_name = dir + "/I2_" + base_name;
  //   
  //   // catch image read/write errors here
  //   try {

  //     // load left and right input image
  //     png::image< png::gray_pixel > left_img(left_img_file_name);
  //     png::image< png::gray_pixel > right_img(right_img_file_name);

  //     // image dimensions
  //     int32_t width  = left_img.get_width();
  //     int32_t height = left_img.get_height();

  //     // convert input images to uint8_t buffer
  //     uint8_t* left_img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
  //     uint8_t* right_img_data = (uint8_t*)malloc(width*height*sizeof(uint8_t));
  //     int32_t k=0;
  //     for (int32_t v=0; v<height; v++) {
  //       for (int32_t u=0; u<width; u++) {
  //         left_img_data[k]  = left_img.get_pixel(u,v);
  //         right_img_data[k] = right_img.get_pixel(u,v);
  //         k++;
  //       }
  //     }

  //     // status
  //     cout << "Processing: Frame: " << i;
  //     
  //     // compute visual odometry
  //     int32_t dims[] = {width,height,width};
  //     if (viso.process(left_img_data,right_img_data,dims)) {
  //     
  //       // on success, update current pose
  //       pose = pose * Matrix::inv(viso.getMotion());
  //     
  //       // output some statistics
  //       double num_matches = viso.getNumberOfMatches();
  //       double num_inliers = viso.getNumberOfInliers();
  //       cout << ", Matches: " << num_matches;
  //       cout << ", Inliers: " << 100.0*num_inliers/num_matches << " %" << ", Current pose: " << endl;
  //       cout << pose << endl << endl;

  //     } else {
  //       cout << " ... failed!" << endl;
  //     }

  //     // release uint8_t buffers
  //     free(left_img_data);
  //     free(right_img_data);

  //   // catch image read errors here
  //   } catch (...) {
  //     cerr << "ERROR: Couldn't read input files!" << endl;
  //     return 1;
  //   }
  // }
  
  // // output
  // cout << "Demo complete! Exiting ..." << endl;

  // // exit
  return 0;
}