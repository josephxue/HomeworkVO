#include "vo.h"

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "motion.h"
#include "feature.h"


std::vector<cv::Point3f> previous_points;
std::vector<cv::KeyPoint> previous_left_keypoints;

cv::Mat previous_left_img;
cv::Mat previous_left_descriptors;


bool ProcessFrame(
    const cv::Mat& left_img, const cv::Mat& right_img, 
    int idx, Eigen::Matrix4d& pose_inc) {

  std::vector<cv::KeyPoint> left_keypoints, right_keypoints;
  cv::Mat left_descriptors, right_descriptors;

  // extract simple orb feature
  FeatureExtraction(left_img,  left_keypoints,  left_descriptors);
  FeatureExtraction(right_img, right_keypoints, right_descriptors);

  // visualize FAST keypoints
  cv::Mat fast_keypoints_visualization;
  cv::drawKeypoints(left_img, left_keypoints, fast_keypoints_visualization);
  cv::imwrite(
      "keypoints" + std::to_string(idx) + ".png",
      fast_keypoints_visualization);
  // cv::imshow("FAST Keypoints Visualization", fast_keypoints_visualization);
  // cv::waitKey(0);

  // stereo matching
  std::vector<cv::DMatch> stereo_matches;
  MatchFeatures(left_descriptors, right_descriptors, stereo_matches);

  // visualize stereo matches
  cv::Mat stereo_matches_visualization;
  cv::drawMatches(left_img, left_keypoints, right_img, right_keypoints, stereo_matches, stereo_matches_visualization);
  cv::imwrite(
      "stereo_matches" + std::to_string(idx) + ".png",
      stereo_matches_visualization);
  // cv::imshow("Stereo Matches Visualization", stereo_matches_visualization);
  // cv::waitKey(0);

  // use disparity to compute depth
  // and inverse project pixels into 3d coordinates
  std::vector<cv::Point3f> points(left_keypoints.size(), cv::Point3f(-1, -1, -1));
  InverseProjection(left_keypoints, right_keypoints, stereo_matches, points);

  bool ret = false;

  if (!previous_points.empty()) {
    ret = true;

    // temporal matching
    std::vector<cv::DMatch> temporal_matches;
    MatchFeatures(previous_left_descriptors, left_descriptors, temporal_matches);

    // visualize temporal matches
    cv::Mat temporal_matches_visualization;
    cv::drawMatches(previous_left_img, previous_left_keypoints, left_img, left_keypoints, temporal_matches, temporal_matches_visualization);
    cv::imwrite(
        "temporal_matches" + std::to_string(idx) + ".png",
        temporal_matches_visualization);
    // cv::imshow("Temporal Matches Visualization", temporal_matches_visualization);
    // cv::waitKey(0);

    // compute pose increment using PnP
    pose_inc = PnpEstimateMotion(previous_points, left_keypoints, temporal_matches);
  }

  previous_points = points;
  previous_left_img = left_img;
  previous_left_keypoints = left_keypoints;
  previous_left_descriptors = left_descriptors;

  return ret;
}