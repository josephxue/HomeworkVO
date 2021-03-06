#include "feature.h"

#include <math.h>

#include <opencv2/xfeatures2d.hpp>


void FeatureExtraction(
    const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {

  // detect FAST keypoints
  DetectKeyPoints(img, keypoints);

  // compute BRIEF descriptor (use OpenCV API)
  cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = 
      cv::xfeatures2d::BriefDescriptorExtractor::create();
  brief->compute(img, keypoints, descriptors);
}


void MatchFeatures(
    const cv::Mat& descriptors1, const cv::Mat& descriptors2, 
    std::vector<cv::DMatch>& matches) {
  
  // do coarse matching 
  std::vector<cv::DMatch> coarse_matches;
  for (int i = 0; i < descriptors1.rows; i++) {

    // brutal force match
    double min_hamming_dist = 10000; int min_j;
    for (int j = 0; j < descriptors2.rows; j++) {
      double hamming_dist = ComputeHammingDistance(descriptors1, i, descriptors2, j);

      if (hamming_dist < min_hamming_dist) {
        min_hamming_dist = hamming_dist;
        min_j = j;
      }
    }
    coarse_matches.emplace_back(cv::DMatch(i, min_j, -1, min_hamming_dist));
  }

  // choose good matching
  std::sort(coarse_matches.begin(), coarse_matches.end());
  double min_dist = coarse_matches[0].distance;
    
  for (auto m : coarse_matches) {
    if (m.distance <= std::max(2*min_dist, 15.0))
      matches.emplace_back(m);
  }
}


void DetectKeyPoints(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints) {
  // compute FAST response
  cv::Mat response = ComputeResponse(img, kThreshold);

  cv::Mat is_localmax = cv::Mat::zeros(cv::Size(response.cols, response.rows), CV_8U);
  NonMaximumSuppression(response, is_localmax, kWindowSize);

  // find local max position to be FAST position
  std::vector<cv::Point> fast_locations;
  cv::findNonZero(is_localmax, fast_locations);

  for (const auto& fl: fast_locations) {
    keypoints.emplace_back(cv::KeyPoint(cv::Point(fl.x, fl.y), -1, -1, response.at<float>(fl)));
  }

  ComputeOrientation(img, keypoints, kWindowSize);
}


cv::Mat ComputeResponse(const cv::Mat& img, int threshold) {
  cv::Mat response = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_32F);
  int intensity_diffs[16];

  for (int v = 3; v < img.rows-3; v++) {
    for (int u = 3; u < img.cols-3; u++) {
      int intensity_center = img.at<uchar>(v, u);
      intensity_diffs[0]  = img.at<uchar>(v-3, u)   - intensity_center;
      intensity_diffs[1]  = img.at<uchar>(v-3, u+1) - intensity_center;
      intensity_diffs[2]  = img.at<uchar>(v-2, u+2) - intensity_center;
      intensity_diffs[3]  = img.at<uchar>(v-1, u+3) - intensity_center;
      intensity_diffs[4]  = img.at<uchar>(v,   u+3) - intensity_center;
      intensity_diffs[5]  = img.at<uchar>(v+1, u+3) - intensity_center;
      intensity_diffs[6]  = img.at<uchar>(v+2, u+2) - intensity_center;
      intensity_diffs[7]  = img.at<uchar>(v+3, u+1) - intensity_center;
      intensity_diffs[8]  = img.at<uchar>(v+3, u)   - intensity_center;
      intensity_diffs[9]  = img.at<uchar>(v+3, u-1) - intensity_center;
      intensity_diffs[10] = img.at<uchar>(v+2, u-2) - intensity_center;
      intensity_diffs[11] = img.at<uchar>(v+1, u-3) - intensity_center;
      intensity_diffs[12] = img.at<uchar>(v,   u-3) - intensity_center;
      intensity_diffs[13] = img.at<uchar>(v-1, u-3) - intensity_center;
      intensity_diffs[14] = img.at<uchar>(v-2, u-2) - intensity_center;
      intensity_diffs[15] = img.at<uchar>(v-3, u-1) - intensity_center;

      bool is_fast_keypoint = false;

      for (int i = 0; i < 16; i++) {
        int j = 0;
        for (; j < 12; j++) {
          if (intensity_diffs[(i+j)%16] <= threshold) {
            break;
          }
        }
        if (j == 12) {
          is_fast_keypoint = true;
          break;
        }
      }

      if (is_fast_keypoint == false) {
        for (int i = 0; i < 16; i++) {
          int j = 0;
          for (; j < 12; j++) {
            if (intensity_diffs[(i+j)%16] >= -threshold) {
              break;
            }
          }
          if (j == 12) {
            is_fast_keypoint = true;
            break;
          }
        }
      }

      float one_response = 0;
      if (is_fast_keypoint == true) {
        for (int i = 0; i < 16; i++) {
          one_response += std::abs(intensity_diffs[i]);
        }
        response.at<float>(v,u) = one_response;
      }
    }
  }

  return response;
}


void NonMaximumSuppression(cv::Mat& response, cv::Mat& is_localmax, int window_size) {
  cv::Mat local_maxes = cv::Mat::zeros(cv::Size(response.cols, response.rows), CV_32F);
  for (int i = window_size/2; i < response.rows-window_size/2; i++) {
    for (int j = window_size/2; j < response.cols-window_size/2; j++) {
      if (response.at<float>(i, j) > 0) {
        double local_max = 0;
        cv::Mat block = response(cv::Rect(j-window_size/2, i-window_size/2, window_size, window_size));
        cv::minMaxLoc(block, NULL, &local_max);
        local_maxes.at<float>(i, j) = local_max;
      }
    }
  }

  for (int i = window_size/2; i < response.rows-window_size/2; i++) {
    for (int j = window_size/2; j < response.cols-window_size/2; j++) {
      if (response.at<float>(i, j) > 0) {
        is_localmax.at<uint8_t>(i,j) = local_maxes.at<float>(i,j) == response.at<float>(i,j) ? 255 : 0;
      }
    }
  }
}


void ComputeOrientation(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, int window_size) {
  int m01 = 0, m10 = 0;
  for (auto kpt : keypoints) {
    for (int i = kpt.pt.x - window_size/2; i <= kpt.pt.y + window_size; i++) {
      for (int j = kpt.pt.x - window_size/2; j <= kpt.pt.x + window_size; j++) {
        m10 += j * img.at<uint8_t>(i,j);
        m01 += i * img.at<uint8_t>(i,j);
      }
    }
    kpt.angle = atan((float)m01/m10);
  }
}


double ComputeHammingDistance(
    const cv::Mat& descriptors1, int i, 
    const cv::Mat& descriptors2, int j) {

  double ret = 0;
  for (int k = 0; k < 32; k++) {
    unsigned char uchar1 = descriptors1.at<uchar>(i,k);
    unsigned char uchar2 = descriptors2.at<uchar>(j,k);
    while (uchar1!= uchar2) {
      // XOR on lowest bit
      ret += (uchar1 & 1) ^ (uchar2 & 1);
      uchar1 >>= 1; 
      uchar2 >>= 1;
    }
  }
  return ret;
}