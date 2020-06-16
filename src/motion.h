#ifndef MOTION_H
#define MOTION_H

#include <opencv2/opencv.hpp>

#include <Eigen/Core>


void InverseProjection(
    const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch> matches, std::vector<cv::Point3f>& points);


Eigen::Matrix4d PnpEstimateMotion(
    const std::vector<cv::Point3f>& points, 
    const std::vector<cv::KeyPoint> keypoints, 
    const std::vector<cv::DMatch> matches);


#endif