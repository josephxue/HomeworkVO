#include "motion.h"

#include <Eigen/SVD>
#include <Eigen/Dense>

#include "params.h"


extern Params params;


void InverseProjection(
    const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch> matches, std::vector<cv::Point3f>& points) {

  for (int i = 0; i < matches.size(); i++) {
    double disparity = std::max(
        keypoints1[matches[i].queryIdx].pt.x - keypoints2[matches[i].trainIdx].pt.x, 
        0.0001f);

    float z = params.K.fx * params.baseline / disparity;
    float x = (keypoints1[matches[i].queryIdx].pt.x - params.K.cu) / params.K.fx * z;
    float y = (keypoints1[matches[i].queryIdx].pt.y - params.K.cv) / params.K.fy * z;

    if (z < 80)
      points[matches[i].queryIdx] = cv::Point3f(x, y, z);
  }
}


Eigen::Matrix4d PnpEstimateMotion(
    const std::vector<cv::Point3f>& points, 
    const std::vector<cv::KeyPoint> keypoints, 
    const std::vector<cv::DMatch> matches) {

  std::vector<cv::Point3f> pts_3d;
  std::vector<cv::Point2f> pts_2d;

  for (auto m : matches) {
    cv::Point3f pt_3d = points[m.queryIdx];
    if (pt_3d.z > 0) {
      pts_3d.emplace_back(pt_3d);
      pts_2d.emplace_back(keypoints[m.trainIdx].pt);
    }
  }

  cv::Mat K = cv::Mat::eye(cv::Size(3,3), CV_32F);
  K.at<float>(0,0) = params.K.fx;
  K.at<float>(0,2) = params.K.cu;
  K.at<float>(1,1) = params.K.fy;
  K.at<float>(1,2) = params.K.cv;

  cv::Mat r, t;
  cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false);
  std::cout << t << std::endl;

  cv::Mat R;
  cv::Rodrigues(r, R);

  Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
  ret(0,0) = R.at<float>(0,0); ret(0,1) = R.at<float>(0,1); ret(0,2) = R.at<float>(0,2); ret(0,3) = t.at<float>(0,0);
  ret(1,0) = R.at<float>(1,0); ret(1,1) = R.at<float>(1,1); ret(1,2) = R.at<float>(1,2); ret(1,3) = t.at<float>(1,0);
  ret(2,0) = R.at<float>(2,0); ret(2,1) = R.at<float>(2,1); ret(2,2) = R.at<float>(2,2); ret(2,3) = t.at<float>(2,0);

  return ret;
}