#include "orb.h"


std::vector<FastKeyPoint> keypoints;


void ComputeOrbFeatures(cv::Mat* img) {
  DetectKeyPoints(img);
}

void DetectKeyPoints(cv::Mat* img) {
  // compute FAST response
  cv::Mat response = ComputeResponse(img, kThreshold);

  cv::Mat local_max;
  cv::dilate(response, local_max, cv::Mat());

  double max_val(0.0);
  cv::minMaxLoc(local_max, NULL, &max_val);

  cv::Mat is_strong_and_local_max = cv::Mat::zeros(cv::Size(response.cols, response.rows), CV_8U);
  std::vector<cv::Point> max_locations;

  float one_response, one_local_max;
  cv::Point max_location;
  for (int i = 0; i < response.rows; i++) {
    for (int j = 0; j < response.cols; j++) {
      one_response  = response.at<float>(i, j);
      one_local_max = local_max.at<float>(i, j);
      is_strong_and_local_max.at<uint8_t>(i, j) = one_response > max_val*kQualityLevel && one_response == one_local_max ? 255 : 0;
    }
  }
  cv::findNonZero(is_strong_and_local_max, max_locations);

  for (const auto& point : max_locations) {
    keypoints.emplace_back(FastKeyPoint(point.x, point.y, response.at<float>(point)));
  }

  // cv::Mat img1 = *img;
  // for (auto p : keypoints) {
  //   cv::circle(img1, cv::Point(p.x, p.y), 3, cv::Scalar(0, 0, 255));
  // }
  // cv::imshow("aaa", img1);
  // cv::waitKey(0);

  keypoints.clear();
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