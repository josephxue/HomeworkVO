#ifndef VO_H
#define VO_H

#include <opencv2/opencv.hpp>

#include <Eigen/Core>


bool ProcessFrame(
    const cv::Mat& left_img, const cv::Mat& right_img,
    Eigen::Matrix4d& pose_inc);


#endif