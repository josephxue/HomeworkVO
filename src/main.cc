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


const int kImageHeight = 192;
const int kImageWidth  = 640;

Params params;
std::vector<Eigen::Matrix4d> poses;


int main (int argc, char** argv) {

  std::string sequence_directory = "../dataset";

  // set camera parameters
  params.K.fx = 371.78;
  params.K.fy = 369.43;
  params.K.cu = 314.05;
  params.K.cv = 88.49;
  params.baseline = 0.54;

  // pose from camera to world
  Eigen::Matrix4d pose = Eigen::MatrixXd::Identity(4,4);
  poses.emplace_back(pose);

  for (int i = 0; i < 50; i++) {
    // image file name
    char img_name[256]; sprintf(img_name, "%06d.png", i);
    std::string left_img_name  = sequence_directory + "/image_0/" + img_name;
    std::string right_img_name = sequence_directory + "/image_1/" + img_name;

    cv::Mat left_img  = cv::imread(left_img_name,  CV_8UC1);
    cv::Mat right_img = cv::imread(right_img_name, CV_8UC1);

    cv::resize(left_img,  left_img,  cv::Size(kImageWidth, kImageHeight));
    cv::resize(right_img, right_img, cv::Size(kImageWidth, kImageHeight));

    if (left_img.empty() || right_img.empty()) {
      std::cerr << "Read images failed!" << std::endl;
      return 2;
    }

    Eigen::Matrix4d pose_inc = Eigen::MatrixXd::Identity(4,4);

    // entry to algorithm
    if (ProcessFrame(left_img, right_img, i, pose_inc)) {
      pose = pose * pose_inc.inverse();
      poses.emplace_back(pose);
    }
  }

  // write result into file
  std::ofstream outfile("result.txt", std::ios::out);
  if (!outfile.is_open()) {
    std::cerr << "Open file failed!" << std::endl;
  }

  for (auto p : poses) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4; j++) {
        outfile << p(i,j) << " ";
      }
    }
    outfile << std::endl;
  }

  outfile.close();

  return 0;
}