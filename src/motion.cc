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

    // whether point has depth through stereo matching
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
  // use OpenCV API
  cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false);

  cv::Mat R;
  // transfer rotation vector to SO3(use OpenCV API)
  cv::Rodrigues(r, R);

  // transfer cv::Mat to Eigen::Matrix4d
  Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
  ret(0,0) = R.at<float>(0,0); ret(0,1) = R.at<float>(0,1); ret(0,2) = R.at<float>(0,2); ret(0,3) = t.at<float>(0,0);
  ret(1,0) = R.at<float>(1,0); ret(1,1) = R.at<float>(1,1); ret(1,2) = R.at<float>(1,2); ret(1,3) = t.at<float>(1,0);
  ret(2,0) = R.at<float>(2,0); ret(2,1) = R.at<float>(2,1); ret(2,2) = R.at<float>(2,2); ret(2,3) = t.at<float>(2,0);

  return ret;
}


Eigen::Matrix4d ICP(
    const std::vector<cv::Point3f>& previous_points, 
    const std::vector<cv::Point3f>& current_points,
    const std::vector<cv::DMatch>& matches, const std::vector<int>& active_idxs) {

  cv::Point3f previous_centroid, current_centroid;

  cv::DMatch active_match;
  for (auto idx : active_idxs) {
    active_match = matches[idx];
    if (previous_points[(int)active_match.queryIdx].z > 0 &&
        current_points[(int)active_match.trainIdx].z > 0) {
      previous_centroid += previous_points[(int)active_match.queryIdx];
      current_centroid  += current_points[(int)active_match.trainIdx];
    }
  }

  previous_centroid /= (int)active_idxs.size();
  current_centroid  /= (int)active_idxs.size();

  std::vector<cv::Point3f> previous_q, current_q;
  for (auto idx : active_idxs) {
    active_match = matches[idx];
    if (previous_points[(int)active_match.queryIdx].z > 0 &&
        current_points[(int)active_match.trainIdx].z > 0) {
      previous_q.emplace_back(previous_points[active_match.queryIdx] - previous_centroid);
      current_q.emplace_back(current_points[active_match.trainIdx]   - current_centroid);
    }
  }

  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for (int i = 0; i < active_idxs.size(); i++) {
    W += Eigen::Vector3d(previous_q[i].x, previous_q[i].y, previous_q[i].z) * 
         Eigen::Vector3d(current_q[i].x,  current_q[i].y,  current_q[i].z).transpose();
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
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

  Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
  ret.block<3,3>(0,0) = R;
  ret.block<3,1>(0,3) = t;

  return ret;
}


std::vector<int> GetRandomSample(int N, int num) {
  // init sample and totalset
  std::vector<int> sample;
  std::vector<int> totalset;

  // create vector containing all indices
  for (int i = 0; i < N; i++)
    totalset.push_back(i);

  // add num indices to current sample
  sample.clear();
  for (int i = 0; i < num; i++) {
    int j = rand() % totalset.size();
    sample.push_back(totalset[j]);
    totalset.erase(totalset.begin()+j);
  }

  // return sample
  return sample;
}


int GetInliersNum(
    const std::vector<cv::Point3f>& previous_points, 
    const std::vector<cv::Point3f>& current_points,
    const std::vector<cv::DMatch>& matches, const Eigen::Matrix4d& pose) {

  int ret = 0;

  Eigen::Matrix3d R = pose.block<3,3>(0,0);
  Eigen::Vector3d t = pose.block<3,1>(0,3);

  for (auto m : matches) {
    cv::Point3f previous_point = previous_points[m.queryIdx];
    cv::Point3f current_point  = current_points[m.trainIdx];

    if (previous_point.z < 0 || current_point.z < 0) continue;

    Eigen::Vector3d current_point_vec  = Eigen::Vector3d(current_point.x,  current_point.y,  current_point.z);
    Eigen::Vector3d previous_point_vec = Eigen::Vector3d(previous_point.x, previous_point.y, previous_point.z);

    Eigen::Vector3d prediction = R * previous_point_vec + t;

    if ((prediction - current_point_vec).norm() < 5) {
      ret++;
    }
  }

  return ret;
}


Eigen::Matrix4d RANSACEstimateMotion(
    const std::vector<cv::Point3f>& previous_points, 
    const std::vector<cv::Point3f>& current_points,
    const std::vector<cv::DMatch>& matches) {

  int N = matches.size();

  Eigen::Matrix4d pose;
  int max_inliers_num = 0;

  for (int i = 0; i < 200; i++) {
    std::vector<int> active_idxs = GetRandomSample(N, N/2);

    Eigen::Matrix4d icp_pose_inc = ICP(previous_points, current_points, matches, active_idxs);

    int inliers_num = GetInliersNum(previous_points, current_points, matches, icp_pose_inc);

    if (inliers_num > max_inliers_num) {
      max_inliers_num = inliers_num;
      pose = icp_pose_inc;
    }
  }

  return pose;
}