//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

void apply_sobel_hypot(float *dst, const uint8_t *src, float scale, int w_, int h_, int thd_per_blk);

void apply_sobel_filter_hypot(float *out_gradient, const uint8_t *in, float scale, int image_width, int image_height, int8_t *sobel_kernel_x, int8_t *sobel_kernel_y,
                              sycl::nd_item<3> item_ct1);

void apply_sobel_sum(float *dst, const uint8_t *src, float scale, int w_, int h_, int thd_per_blk);

void apply_sobel_filter_sum(float *out_gradient, const uint8_t *in, float scale, int image_width, int image_height, int8_t *sobel_kernel_x, int8_t *sobel_kernel_y,
                            sycl::nd_item<3> item_ct1);
