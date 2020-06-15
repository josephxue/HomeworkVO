#ifndef MOTION_H
#define MOTION_H

#include <opencv2/opencv.hpp>

#include <Eigen/Core>


void InverseProjection(
    const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch> matches, std::unordered_map<int, cv::Point3f>& points);

Eigen::Matrix4d EstimateMotion(
    std::unordered_map<int, cv::Point3f>& previous_points, 
    std::unordered_map<int, cv::Point3f>& current_points,
    const std::vector<cv::DMatch>& matches);

#endif