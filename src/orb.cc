#include "orb.h"

#include <math.h>


std::vector<FastKeyPoint> keypoints;


void ComputeOrbFeatures(cv::Mat* img) {
  DetectKeyPoints(img);

  ComputeOrientation(img, keypoints, kWindowSize);
}

void DetectKeyPoints(cv::Mat* img) {
  // compute FAST response
  cv::Mat response = ComputeResponse(img, kThreshold);
  
  cv::Mat is_localmax = cv::Mat::zeros(cv::Size(response.cols, response.rows), CV_8U);
  NonMaximumSuppression(response, is_localmax, kWindowSize);
  std::vector<cv::Point> fast_locations;

  cv::findNonZero(is_localmax, fast_locations);

  for (const auto& fl: fast_locations) {
    keypoints.emplace_back(FastKeyPoint(fl.x, fl.y, response.at<float>(fl)));
  }

  // cv::Mat fast_keypoints_visualization = *img;
  // cvtColor(fast_keypoints_visualization, fast_keypoints_visualization, CV_GRAY2BGR);
  // for (auto p : keypoints) {
  //   cv::circle(fast_keypoints_visualization, cv::Point(p.x, p.y), 2, cv::Scalar(0, 255, 0));
  // }
  // cv::imshow("FAST Keypoints Visualization", fast_keypoints_visualization);
  // cv::waitKey(0);
}

cv::Mat ComputeResponse(cv::Mat* img, int threshold) {
  cv::Mat response = cv::Mat::zeros(cv::Size((*img).cols, (*img).rows), CV_32F);
  int intensity_diffs[16];

  for (int v = 3; v < (*img).rows-3; v++) {
    for (int u = 3; u < (*img).cols-3; u++) {
      int intensity_center = (*img).at<uchar>(v, u);
      intensity_diffs[0]  = (*img).at<uchar>(v-3, u)   - intensity_center;
      intensity_diffs[1]  = (*img).at<uchar>(v-3, u+1) - intensity_center;
      intensity_diffs[2]  = (*img).at<uchar>(v-2, u+2) - intensity_center;
      intensity_diffs[3]  = (*img).at<uchar>(v-1, u+3) - intensity_center;
      intensity_diffs[4]  = (*img).at<uchar>(v,   u+3) - intensity_center;
      intensity_diffs[5]  = (*img).at<uchar>(v+1, u+3) - intensity_center;
      intensity_diffs[6]  = (*img).at<uchar>(v+2, u+2) - intensity_center;
      intensity_diffs[7]  = (*img).at<uchar>(v+3, u+1) - intensity_center;
      intensity_diffs[8]  = (*img).at<uchar>(v+3, u)   - intensity_center;
      intensity_diffs[9]  = (*img).at<uchar>(v+3, u-1) - intensity_center;
      intensity_diffs[10] = (*img).at<uchar>(v+2, u-2) - intensity_center;
      intensity_diffs[11] = (*img).at<uchar>(v+1, u-3) - intensity_center;
      intensity_diffs[12] = (*img).at<uchar>(v,   u-3) - intensity_center;
      intensity_diffs[13] = (*img).at<uchar>(v-1, u-3) - intensity_center;
      intensity_diffs[14] = (*img).at<uchar>(v-2, u-2) - intensity_center;
      intensity_diffs[15] = (*img).at<uchar>(v-3, u-1) - intensity_center;

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

void ComputeOrientation(cv::Mat* img, std::vector<FastKeyPoint>& keypoints, int window_size) {
  int m01 = 0, m10 = 0;
  for (auto pt : keypoints) {
    for (int i = pt.y - window_size/2; i <= pt.y + window_size; i++) {
      for (int j = pt.x - window_size/2; j <= pt.x + window_size; j++) {
        m10 += j * (*img).at<uint8_t>(i,j);
        m01 += i * (*img).at<uint8_t>(i,j);
      }
    }
    pt.orientation = atan((float)m01/m10);
  }
}