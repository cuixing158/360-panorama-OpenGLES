//
// Created by 1 on 2024/8/1.
//

#ifndef MY360PANORAMA_OPENCV_UTILS_H
#define MY360PANORAMA_OPENCV_UTILS_H

#include <opencv2/core.hpp>
using namespace cv;
void myFlip(Mat& src);
void myBlur(Mat& src, float sigma);

#endif //MY360PANORAMA_OPENCV_UTILS_H
