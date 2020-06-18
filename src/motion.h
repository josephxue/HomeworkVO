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


Eigen::Matrix4d ICP(
    const std::vector<cv::Point3f>& previous_points, 
    const std::vector<cv::Point3f>& current_points,
    const std::vector<cv::DMatch>& matches, const std::vector<int>& active_idxs);


std::vector<int> GetRandomSample(int N, int num);


int GetInliersNum(
    const std::vector<cv::Point3f>& previous_points, 
    const std::vector<cv::Point3f>& current_points,
    const std::vector<cv::DMatch>& matches, const Eigen::Matrix4d& pose);


Eigen::Matrix4d RANSACEstimateMotion(
    const std::vector<cv::Point3f>& previous_points, 
    const std::vector<cv::Point3f>& current_points,
    const std::vector<cv::DMatch>& matches);


#endif