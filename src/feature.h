#ifndef FEATURE_H
#define FEATURE_H


#include <opencv2/opencv.hpp>


const int kWindowSize = 3;

const float kThreshold = 50;

extern std::vector<cv::KeyPoint> keypoints;


void FeatureExtraction(
    const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

void MatchFeatures(
    const cv::Mat& left_descriptors, const cv::Mat& right_descriptors, 
    std::vector<cv::DMatch>& matches);

void DetectKeyPoints(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints);

cv::Mat ComputeResponse(const cv::Mat& img, int threshold);

void NonMaximumSuppression(cv::Mat& response, cv::Mat& is_localmax, int window_size);

void ComputeOrientation(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, int window_size);

double ComputeHammingDistance(
    const cv::Mat& left_descriptors,  int i, 
    const cv::Mat& right_descriptors, int j);

#endif