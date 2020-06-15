#include <stdint.h>

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

#include "vo.h"
#include "params.h"
#include "feature.h"


const bool kVisualize = false;

const int kImageHeight = 192;
const int kImageWidth  = 640;

Params params;


int main (int argc, char** argv) {
  if (argc<2) {
    std::cerr << "Usage: ./HomeworkVO path/to/sequence/00" << std::endl;
    return 1;
  }

  std::string sequence_directory = argv[1];

  std::ofstream outfile("result.txt", std::ios::app);
  if (!outfile.is_open()) {
    std::cerr << "Open file failed!" << std::endl;
  }

  // set camera parameters
  params.K.fx = 370.75;
  params.K.fy = 367.08;
  params.K.cu = 313.15;
  params.K.cv = 94.58;
  params.baseline = 0.54;

  // pose from camera to world
  Eigen::Matrix4d pose = Eigen::MatrixXd::Identity(4,4);

  for (int i = 0; i < 3; i++) {
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

    Eigen::Matrix4d pose_inc;

    // entry to algorithm
    if (ProcessFrame(left_img, right_img, pose_inc)) {
      pose = pose * pose_inc.inverse();

      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
          outfile << pose(i, j) << " ";
        }
      }
      outfile << std::endl;
    }
  }

  outfile.close();

  return 0;
}