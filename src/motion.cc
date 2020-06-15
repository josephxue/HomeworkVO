#include "motion.h"

#include <Eigen/SVD>
#include <Eigen/Dense>

#include "params.h"


extern Params params;


void InverseProjection(
    const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch> matches, std::unordered_map<int, cv::Point3f>& points) {

  for (int i = 0; i < matches.size(); i++) {
    double disparity = std::max(
        keypoints1[matches[i].queryIdx].pt.x - keypoints2[matches[i].trainIdx].pt.x, 
        0.0001f);

    float x = (keypoints1[matches[i].queryIdx].pt.x - params.K.cu) * params.baseline / disparity;
    float y = (keypoints1[matches[i].queryIdx].pt.y - params.K.cv) * params.baseline / disparity;
    float z = params.K.fx * params.baseline / disparity;

    if (z < 50)
      points[matches[i].queryIdx] = cv::Point3f(x, y, z);
  }
}


Eigen::Matrix4d EstimateMotion(
    std::unordered_map<int, cv::Point3f>& previous_points, 
    std::unordered_map<int, cv::Point3f>& current_points,
    const std::vector<cv::DMatch>& matches) {
  
  cv::Point3f previous_centroid, current_centroid;

  for (auto m : matches) {
    previous_centroid += previous_points[(int)m.queryIdx];
    current_centroid  += current_points[(int)m.trainIdx];
  }

  previous_centroid /= (int)matches.size();
  current_centroid  /= (int)matches.size();

  std::vector<cv::Point3f> previous_q, current_q;
  for (auto m : matches) {
    previous_q.emplace_back(previous_points[m.queryIdx] - previous_centroid);
    current_q.emplace_back(current_points[m.trainIdx]   - current_centroid);
  }

  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for (int i = 0; i < previous_q.size(); i++) {
    W += Eigen::Vector3d(previous_q[i].x, previous_q[i].y, previous_q[i].z) *
         Eigen::Vector3d(current_q[i].x,  current_q[i].y,  current_q[i].z).transpose();
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  if (U.determinant() * V.determinant() < 0) {
    for (int x = 0; x < 3; x++) {
      U(x, 2) *= -1;
    }
	}

  Eigen::Matrix3d R = U * (V.transpose());
  Eigen::Vector3d t = Eigen::Vector3d(previous_centroid.x, previous_centroid.y, previous_centroid.z) - 
      R * Eigen::Vector3d(current_centroid.x, current_centroid.y, current_centroid.z);

  Eigen::Matrix4d ret;
  ret.block<3,3>(0,0) = R;
  ret.block<3,1>(0,3) = t;
  ret.block<1,3>(3,0) = Eigen::VectorXd::Zero(1,3);
  ret(3,3) = 1;

  std::cout << ret << std::endl;

  return ret;
}