#ifndef ORB_H
#define ORB_H


#include <opencv2/opencv.hpp>


const int kWindowSize = 3;

const float kThreshold = 50;


struct FastKeyPoint {
  int x, y;
  double response;
  double orientation;

  FastKeyPoint(int _x, int _y, double _response) : x(_x), y(_y), response(_response) {}
};

extern std::vector<FastKeyPoint> keypoints;

void ComputeOrbFeatures(cv::Mat* img);

void DetectKeyPoints(cv::Mat* img);

cv::Mat ComputeResponse(cv::Mat* img, int threshold);

void NonMaximumSuppression(cv::Mat& response, cv::Mat& is_localmax, int window_size);

void ComputeOrientation(cv::Mat* img, std::vector<FastKeyPoint>& keypoints, int window_size);

#endif