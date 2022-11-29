//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

void apply_canny(uint8_t *dst, uint8_t *dst_week, const uint8_t *src, int weak_threshold, int strong_threshold, int w_, int h_, int thd_per_blk);
void apply_gaussian_filter(uint8_t *dst, const uint8_t *src, int image_width, int image_height, float *d_blur_kernel,
                           sycl::nd_item<3> item_ct1);
void apply_sobel_filter(float *out_gradient, uint8_t *out_segment, const uint8_t *in, int image_width, int image_height, int8_t *sobel_kernel_x, int8_t *sobel_kernel_y,
                        sycl::nd_item<3> item_ct1);
// void apply_non_max_suppression(float* out_M, float* in_gradient, uint8_t* in_segment, int image_width, int image_height,
//                                sycl::nd_item<3> item_ct1);
// void apply_double_threshold(uint8_t* dst, uint8_t* dst_week, float* M_, int strong_threshold, int weak_threshold, int image_width, int image_height,
//                             sycl::nd_item<3> item_ct1);
void apply_non_max_suppression(uint8_t *dst, uint8_t *dst_week, float *out_M, float *in_gradient, uint8_t *in_segment, int image_width, int image_height,
                               int strong_threshold, int weak_threshold, sycl::nd_item<3> item_ct1);
// __global__ void apply_edge_hysteresis(uint8_t* out, int* in, int image_width, int image_height);
void waitForKernelCompletion();
