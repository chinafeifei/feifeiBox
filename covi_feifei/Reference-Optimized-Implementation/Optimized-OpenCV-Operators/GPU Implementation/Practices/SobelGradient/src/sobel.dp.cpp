//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#define _USE_MATH_DEFINES
#define KERNEL_SIZE 3
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "sobel.h"
#include <iostream>
#include <math.h>
#include <chrono>

void waitForKernelCompletion()
try
{
    /*
    DPCT1003:21: Migrated API does not return error code. (*, 0) is
    inserted. You may need to rewrite this code.
    */
    dpct::get_current_device().queues_wait_and_throw();
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void apply_sobel_hypot(float *final_pixels, const uint8_t *ori_pixels, float scale, int image_width, int image_height,
                       int thd_per_blk)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();

    const int8_t sobel_kernel_x[] = {-1, 0, 1,
                                     -2, 0, 2,
                                     -1, 0, 1};
    const int8_t sobel_kernel_y[] = {1, 2, 1,
                                     0, 0, 0,
                                     -1, -2, -1};
    /* kernel execution configuration parameters */
    const int num_blks = (image_height * image_width + thd_per_blk - 1) / thd_per_blk;
    ;

    /* device buffers */
    uint8_t *in;
    float *gradient_pixels;
    int8_t *sobel_kernel_x_gpu;
    int8_t *sobel_kernel_y_gpu;

    float elapsed = 0;
    float num_iters = 5;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

    /* allocate device memory */
    in = (uint8_t *)sycl::malloc_device(
        sizeof(uint8_t) * image_height * image_width, q_ct1);
    gradient_pixels = (float *)sycl::malloc_device(
        sizeof(float) * image_height * image_width, q_ct1);
    ;
    sobel_kernel_x_gpu =
        (int8_t *)sycl::malloc_device(sizeof(int8_t) * 3 * 3, q_ct1);
    sobel_kernel_y_gpu =
        (int8_t *)sycl::malloc_device(sizeof(int8_t) * 3 * 3, q_ct1);

    // start_ct1 = std::chrono::steady_clock::now(); // start timer
    /* data transfer image pixels to device */
    q_ct1.memcpy(in, ori_pixels, image_height * image_width * sizeof(uint8_t));
    q_ct1.memcpy(sobel_kernel_x_gpu, sobel_kernel_x,
                 sizeof(int8_t) * KERNEL_SIZE * KERNEL_SIZE);
    q_ct1
        .memcpy(sobel_kernel_y_gpu, sobel_kernel_y,
                sizeof(int8_t) * KERNEL_SIZE * KERNEL_SIZE)
        .wait();

    for (int i = 0; i <= num_iters; i++)
    {
        waitForKernelCompletion();
        start_ct1 = std::chrono::steady_clock::now(); // start timer
        // apply sobel kernels
        q_ct1.submit([&](sycl::handler &cgh)
                     { cgh.parallel_for(
                           sycl::nd_range<3>(sycl::range<3>(1, 1, num_blks) *
                                                 sycl::range<3>(1, 1, thd_per_blk),
                                             sycl::range<3>(1, 1, thd_per_blk)),
                           [=](sycl::nd_item<3> item_ct1)
                           {
                               apply_sobel_filter_hypot(gradient_pixels, in,
                                                        scale, image_width, image_height,
                                                        sobel_kernel_x_gpu, sobel_kernel_y_gpu,
                                                        item_ct1);
                           }); })
            .wait();

        waitForKernelCompletion();
        stop_ct1 = std::chrono::steady_clock::now(); // end timer

        if (i > 0)
            elapsed += std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                           .count();
    }

    /* wait for everything to finish */
    // dev_ct1.queues_wait_and_throw();

    /* copy result back to the host */
    q_ct1.memcpy(final_pixels, gradient_pixels, image_width * image_height * sizeof(float));

    sycl::free(in, q_ct1);
    sycl::free(gradient_pixels, q_ct1);
    sycl::free(sobel_kernel_x_gpu, q_ct1);
    sycl::free(sobel_kernel_y_gpu, q_ct1);

    std::cout << "The sobel hypot elapsed time in gpu was " << elapsed / num_iters << " ms" << std::endl;
}

void apply_sobel_filter_hypot(float *gradient_pixels, const uint8_t *in_pixels, float scale, int image_width, int image_height, int8_t *sobel_kernel_x, int8_t *sobel_kernel_y,
                              sycl::nd_item<3> item_ct1)
{
    // Sobel
    int pixNum = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2);
    if (!(pixNum >= 0 && pixNum < image_height * image_width))
        return;
    int x = pixNum % image_width;
    int y = pixNum / image_width;
    int offset_xy = 1; // 3x3
    // if (x < offset_xy || x >= image_width - offset_xy || y < offset_xy || y >= image_height - offset_xy)
    // 	return;
    float convolve_X = 0.0;
    float convolve_Y = 0.0;
    int k = 0;

    for (int i = -offset_xy; i <= offset_xy; i++)
    {
        for (int j = -offset_xy; j <= offset_xy; j++)
        {
            int kernel_x = x + j;
            int kernel_y = y + i;
            if (kernel_x < 0)
                kernel_x = 0;
            if (kernel_x >= image_width)
                kernel_x = image_width - 1;
            if (kernel_y < 0)
                kernel_y = 0;
            if (kernel_y >= image_height)
                kernel_y = image_height - 1;
            convolve_X += in_pixels[kernel_y * image_width + kernel_x] * sobel_kernel_x[k];
            convolve_Y += in_pixels[kernel_y * image_width + kernel_x] * sobel_kernel_y[k];
            k++;
        }
    }

    // gradient hypot
    if (convolve_X == 0.0 && convolve_Y == 0.0)
    {
        gradient_pixels[pixNum] = 0;
    }
    else
    {
        gradient_pixels[pixNum] = scale * ((sycl::sqrt(
                                              (convolve_X * convolve_X) + (convolve_Y * convolve_Y))));
    }
}

void apply_sobel_sum(float *final_pixels, const uint8_t *ori_pixels, float scale, int image_width, int image_height,
                     int thd_per_blk)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();

    const int8_t sobel_kernel_x[] = {-1, 0, 1,
                                     -2, 0, 2,
                                     -1, 0, 1};
    const int8_t sobel_kernel_y[] = {1, 2, 1,
                                     0, 0, 0,
                                     -1, -2, -1};
    /* kernel execution configuration parameters */
    const int num_blks = (image_height * image_width + thd_per_blk - 1) / thd_per_blk;

    /* device buffers */
    uint8_t *in;
    float *gradient_pixels;
    int8_t *sobel_kernel_x_gpu;
    int8_t *sobel_kernel_y_gpu;

    float elapsed = 0;
    float num_iters = 5;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

    /* allocate device memory */
    in = (uint8_t *)sycl::malloc_device(
        sizeof(uint8_t) * image_height * image_width, q_ct1);
    gradient_pixels = (float *)sycl::malloc_device(
        sizeof(float) * image_height * image_width, q_ct1);
    ;
    sobel_kernel_x_gpu =
        (int8_t *)sycl::malloc_device(sizeof(int8_t) * 3 * 3, q_ct1);
    sobel_kernel_y_gpu =
        (int8_t *)sycl::malloc_device(sizeof(int8_t) * 3 * 3, q_ct1);

    // start_ct1 = std::chrono::steady_clock::now(); // start timer
    /* data transfer image pixels to device */
    q_ct1.memcpy(in, ori_pixels, image_height * image_width * sizeof(uint8_t));
    q_ct1.memcpy(sobel_kernel_x_gpu, sobel_kernel_x,
                 sizeof(int8_t) * KERNEL_SIZE * KERNEL_SIZE);
    q_ct1
        .memcpy(sobel_kernel_y_gpu, sobel_kernel_y,
                sizeof(int8_t) * KERNEL_SIZE * KERNEL_SIZE)
        .wait();

    for (int i = 0; i <= num_iters; i++)
    {
        waitForKernelCompletion();
        start_ct1 = std::chrono::steady_clock::now(); // start timer
        // apply sobel kernels
        q_ct1.submit([&](sycl::handler &cgh)
                     { cgh.parallel_for(
                           sycl::nd_range<3>(sycl::range<3>(1, 1, num_blks) *
                                                 sycl::range<3>(1, 1, thd_per_blk),
                                             sycl::range<3>(1, 1, thd_per_blk)),
                           [=](sycl::nd_item<3> item_ct1)
                           {
                               apply_sobel_filter_sum(gradient_pixels, in,
                                                      scale, image_width, image_height,
                                                      sobel_kernel_x_gpu, sobel_kernel_y_gpu,
                                                      item_ct1);
                           }); })
            .wait();

        waitForKernelCompletion();
        stop_ct1 = std::chrono::steady_clock::now(); // end timer

        if (i > 0)
            elapsed += std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1)
                           .count();
    }

    /* wait for everything to finish */
    // dev_ct1.queues_wait_and_throw();

    /* copy result back to the host */
    q_ct1.memcpy(final_pixels, gradient_pixels, image_width * image_height * sizeof(float));

    sycl::free(in, q_ct1);
    sycl::free(gradient_pixels, q_ct1);
    sycl::free(sobel_kernel_x_gpu, q_ct1);
    sycl::free(sobel_kernel_y_gpu, q_ct1);

    std::cout << "The sobel sum elapsed time in gpu was " << elapsed / num_iters << " ms" << std::endl;
}

void apply_sobel_filter_sum(float *gradient_pixels, const uint8_t *in_pixels, float scale, int image_width, int image_height, int8_t *sobel_kernel_x, int8_t *sobel_kernel_y,
                            sycl::nd_item<3> item_ct1)
{
    // Sobel
    int pixNum = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2);
    if (!(pixNum >= 0 && pixNum < image_height * image_width))
        return;
    int x = pixNum % image_width;
    int y = pixNum / image_width;
    int offset_xy = 1; // 3x3
    // if (x < offset_xy || x >= image_width - offset_xy || y < offset_xy || y >= image_height - offset_xy)
    // 	return;
    float convolve_X = 0.0;
    float convolve_Y = 0.0;
    int k = 0;

    for (int i = -offset_xy; i <= offset_xy; i++)
    {
        for (int j = -offset_xy; j <= offset_xy; j++)
        {
            int kernel_x = x + j;
            int kernel_y = y + i;
            if (kernel_x < 0)
                kernel_x = 0;
            if (kernel_x >= image_width)
                kernel_x = image_width - 1;
            if (kernel_y < 0)
                kernel_y = 0;
            if (kernel_y >= image_height)
                kernel_y = image_height - 1;
            convolve_X += in_pixels[kernel_y * image_width + kernel_x] * sobel_kernel_x[k];
            convolve_Y += in_pixels[kernel_y * image_width + kernel_x] * sobel_kernel_y[k];
            k++;
        }
    }

    // gradient sum
    if (convolve_X == 0.0 && convolve_Y == 0.0)
    {
        gradient_pixels[pixNum] = 0;
    }
    else
    {
        gradient_pixels[pixNum] = scale * ((sycl::abs(convolve_X) + sycl::abs(convolve_Y)));
    }
}