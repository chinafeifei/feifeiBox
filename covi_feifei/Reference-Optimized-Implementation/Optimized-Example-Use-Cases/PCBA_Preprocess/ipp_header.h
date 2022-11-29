//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#pragma once

#include "ippi.h"
#include "opencv2/core.hpp"

void tbb_ipp_boxfilter_11x11(cv::Mat &image, cv::Mat &output);

void ipp_morph_close(cv::Mat &image, cv::Mat &output, cv::Mat &kernel);

void ipp_morph_close_tbb(cv::Mat &image, cv::Mat &output, cv::Mat &kernel);

int resizeIPP_Nearest_C3R(cv::Mat &image, cv::Mat &dst, cv::Size dsize, double inv_scale_x, double inv_scale_y);

int resizeIPP_Linear(cv::Mat &image, cv::Mat &dst, cv::Size dsize, double inv_scale_x = 0, double inv_scale_y = 0);

void resize_ipp_tl(cv::Mat &src, cv::Mat &dst, cv::Size dsize, double inv_scale_x, double inv_scale_y);

void ipp_threshold(cv::Mat &image, cv::Mat &output, Ipp8u threshold);

void ipp_threshold_tbb(cv::Mat &image, cv::Mat &output, Ipp8u threshold);

void ipp_threshold_omp(cv::Mat &image, cv::Mat &output, Ipp8u threshold);

void ipp_threshold_omp_otsu(cv::Mat &image, cv::Mat &output);
