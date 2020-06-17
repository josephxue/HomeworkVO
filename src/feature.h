#ifndef FEATURE_H
#define FEATURE_H


#include <opencv2/opencv.hpp>


const int kWindowSize = 3;
const float kThreshold = 50;


void FeatureExtraction(
    const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, 
    cv::Mat& descriptors);


void DetectKeyPoints(
    const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints);


cv::Mat ComputeResponse(const cv::Mat& img, int threshold);


void NonMaximumSuppression(
    cv::Mat& response, cv::Mat& is_localmax, int window_size);


void ComputeOrientation(
    const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, 
    int window_size);


void MatchFeatures(
    const cv::Mat& descriptors1, const cv::Mat& descriptors2, 
    std::vector<cv::DMatch>& matches);


double ComputeHammingDistance(
    const cv::Mat& descriptors1, int i, 
    const cv::Mat& descriptors2, int j);


#endif