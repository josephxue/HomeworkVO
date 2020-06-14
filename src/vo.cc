#include "vo.h"

#include <iostream>

#include <opencv2/xfeatures2d.hpp>

#include "feature.h"


void ProcessFrame(cv::Mat* left_img, cv::Mat* right_img) {
  // detect FAST keypoints
  std::vector<cv::KeyPoint> left_keypoints, right_keypoints;
  DetectKeyPoints(left_img,  left_keypoints);
  DetectKeyPoints(right_img, right_keypoints);

  // compute BRIEF descriptor (use OpenCV API)
  cv::Mat left_descriptors, right_descriptors;
  cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = 
      cv::xfeatures2d::BriefDescriptorExtractor::create();
  brief->compute(*left_img,  left_keypoints,  left_descriptors);
  brief->compute(*right_img, right_keypoints, right_descriptors);

  // do coarse matching 
  std::vector<cv::DMatch> matches;
  for (int i = 0; i < left_descriptors.rows; i++) {
    double min_hamming_dist = 10000; int min_j;
    for (int j = 0; j < right_descriptors.rows; j++) {
      double hamming_dist = ComputeHammingDistance(left_descriptors, i, right_descriptors, j);

      if (hamming_dist < min_hamming_dist) {
        min_hamming_dist = hamming_dist;
        min_j = j;
      }
    }
    matches.emplace_back(cv::DMatch(i, min_j, -1, min_hamming_dist));
  }

  // choose good matching
  std::sort(matches.begin(), matches.end());
  double min_dist = matches[0].distance;
    
  std::vector<cv::DMatch> good_matches;
  for (auto m : matches) {
    if (m.distance <= std::max(2*min_dist, 30.0))
      good_matches.emplace_back(m);
  }

  // cv::Mat matches_visualization;
  // cv::drawMatches(*left_img, left_keypoints, *right_img, right_keypoints, matches, matches_visualization);
  // cv::imshow("Matches Visualization", matches_visualization);
  // cv::waitKey(0);

  // cv::Mat good_matches_visualization;
  // cv::drawMatches(*left_img, left_keypoints, *right_img, right_keypoints, good_matches, good_matches_visualization);
  // cv::imshow("Good Matches Visualization", good_matches_visualization);
  // cv::waitKey(0);
}