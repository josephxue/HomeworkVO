#include "vo.h"

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "motion.h"
#include "feature.h"


std::vector<cv::Point3f> previous_points;
std::vector<cv::KeyPoint> previous_left_keypoints, previous_right_keypoints;

cv::Mat previous_left_img;
cv::Mat previous_left_descriptors, previous_right_descriptors;


bool ProcessFrame(
    const cv::Mat& left_img, const cv::Mat& right_img, 
    Eigen::Matrix4d& pose_inc) {

  std::vector<cv::KeyPoint> left_keypoints, right_keypoints;
  cv::Mat left_descriptors, right_descriptors;

  FeatureExtraction(left_img,  left_keypoints,  left_descriptors);
  FeatureExtraction(right_img, right_keypoints, right_descriptors);

  // cv::Mat fast_keypoints_visualization;
  // cv::drawKeypoints(left_img, keypoints, fast_keypoints_visualization);
  // cv::imshow("FAST Keypoints Visualization", fast_keypoints_visualization);
  // cv::waitKey(0);

  std::vector<cv::DMatch> stereo_matches;
  MatchFeatures(left_descriptors, right_descriptors, stereo_matches);

  // cv::Mat stereo_matches_visualization;
  // cv::drawMatches(left_img, left_keypoints, right_img, right_keypoints, stereo_matches, stereo_matches_visualization);
  // cv::imshow("Stereo Matches Visualization", stereo_matches_visualization);
  // cv::waitKey(0);

  std::vector<cv::Point3f> points(left_keypoints.size(), cv::Point3f(-1, -1, -1));
  InverseProjection(left_keypoints, right_keypoints, stereo_matches, points);

  bool ret = false;

  if (!previous_points.empty()) {
    ret = true;

    std::vector<cv::DMatch> temporal_matches;
    MatchFeatures(left_descriptors, previous_left_descriptors, temporal_matches);

    // cv::Mat temporal_matches_visualization;
    // cv::drawMatches(left_img, left_keypoints, previous_left_img, previous_left_keypoints, temporal_matches, temporal_matches_visualization);
    // cv::imshow("Temporal Matches Visualization", temporal_matches_visualization);
    // cv::waitKey(0);

    pose_inc = RANSACEstimateMotion(previous_points, points, temporal_matches);
  }

  previous_points = points;

  previous_left_img = left_img;

  previous_left_keypoints  = left_keypoints;
  previous_right_keypoints = right_keypoints;

  previous_left_descriptors  = left_descriptors;
  previous_right_descriptors = right_descriptors;

  return ret;
}