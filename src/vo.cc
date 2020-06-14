#include "vo.h"

#include <iostream>


#include "feature.h"


void ProcessFrame(const cv::Mat& left_img, const cv::Mat& right_img) {
  std::vector<cv::KeyPoint> left_keypoints, right_keypoints;
  cv::Mat left_descriptors, right_descriptors;

  FeatureExtraction(left_img,  left_keypoints,  left_descriptors);
  FeatureExtraction(right_img, right_keypoints, right_descriptors);

  // cv::Mat fast_keypoints_visualization;
  // cv::drawKeypoints(left_img, keypoints, fast_keypoints_visualization);
  // cv::imshow("FAST Keypoints Visualization", fast_keypoints_visualization);
  // cv::waitKey(0);

  std::vector<cv::DMatch> matches;
  MatchFeatures(left_descriptors, right_descriptors, matches);

  cv::Mat matches_visualization;
  cv::drawMatches(left_img, left_keypoints, right_img, right_keypoints, matches, matches_visualization);
  cv::imshow("Matches Visualization", matches_visualization);
  cv::waitKey(0);
}