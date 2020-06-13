#ifndef ORB_H
#define ORB_H


#include <opencv2/opencv.hpp>


const float kThreshold = 50;
const float kQualityLevel = 0.01;


struct FastKeyPoint {
  int x, y;
  double response;

  FastKeyPoint(int _x, int _y, double _response) : x(_x), y(_y), response(_response) {}
};

extern std::vector<FastKeyPoint> keypoints;

void ComputeOrbFeatures(cv::Mat* img);

void DetectKeyPoints(cv::Mat* img);

cv::Mat ComputeResponse(cv::Mat* img, int threshold);

#endif