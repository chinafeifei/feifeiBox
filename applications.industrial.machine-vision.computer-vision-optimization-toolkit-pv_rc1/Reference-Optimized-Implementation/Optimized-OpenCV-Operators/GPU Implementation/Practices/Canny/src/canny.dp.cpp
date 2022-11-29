//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#define _USE_MATH_DEFINES
#define KERNEL_SIZE 3
#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <dpct/dpct.hpp>
#include "canny.h"
#include <iostream>
#include <math.h>
#include <chrono>
#include "esimd_test_utils.hpp"

#define GET_BLOCK_SIZE(t, blk) (t + blk - 1) / blk

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

void apply_canny(uint8_t *final_pixels, uint8_t *week_pixels,
                 const uint8_t *ori_pixels, int weak_threshold,
                 int strong_threshold, int image_width, int image_height,
                 int thd_per_blk)
{
    //  dpct::device_ext &dev_ct1 = dpct::get_current_device();
    //  sycl::queue &q_ct1 = dev_ct1.default_queue();

    sycl::queue q_ct1(gpu_selector{}, esimd_test::createExceptionHandler(), property::queue::enable_profiling{});

    // gaussian kernel
    const float gaussian_kernel[3] = {
        0.25, 0.5, 0.25};
    const int8_t sobel_kernel_x[] = {-1, 0, 1,
                                     -2, 0, 2,
                                     -1, 0, 1};
    const int8_t sobel_kernel_y[] = {1, 2, 1,
                                     0, 0, 0,
                                     -1, -2, -1};
    /* kernel execution configuration parameters */
    const int num_blks = (image_height * image_width + thd_per_blk - 1) / thd_per_blk;
    const int grid = 0;

    /* device buffers */
    uint8_t *in_gaussian, *out, *out_gaussian, *out_week;
    float *gradient_pixels;
    float *max_pixels;
    uint8_t *segment_pixels;
    // float* gaussian_kernel_gpu;
    int8_t *sobel_kernel_x_gpu;
    int8_t *sobel_kernel_y_gpu;

    float elapsed = 0;
    float elapsed1 = 0;
    float elapsed2 = 0;
    float elapsed3 = 0;
    unsigned num_iters = 5;

    sycl::event start, stop;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct2;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct3;
    /*
    DPCT1026:0: The call to cudaEventCreate was removed because this call is
    redundant in DPC++.
    */
    /*
    DPCT1026:1: The call to cudaEventCreate was removed because this call is
    redundant in DPC++.
    */

    out_gaussian = (uint8_t *)sycl::aligned_alloc_shared(4096,
                                                         sizeof(uint8_t) * image_height * image_width, q_ct1);
    in_gaussian = (uint8_t *)sycl::aligned_alloc_shared(4096,
                                                        sizeof(uint8_t) * image_height * image_width, q_ct1);
    /* allocate device memory */
    out = (uint8_t *)sycl::malloc_device(
        sizeof(uint8_t) * image_height * image_width, q_ct1);
    gradient_pixels = (float *)sycl::malloc_device(
        sizeof(float) * image_height * image_width, q_ct1);
    out_week = (uint8_t *)sycl::malloc_device(
        sizeof(uint8_t) * image_height * image_width, q_ct1);
    max_pixels = (float *)sycl::malloc_device(
        sizeof(float) * image_height * image_width, q_ct1);
    segment_pixels = (uint8_t *)sycl::malloc_device(
        sizeof(uint8_t) * image_height * image_width, q_ct1);
    // gaussian_kernel_gpu = (float *)sycl::malloc_shared(
    //     sizeof(float) * KERNEL_SIZE, q_ct1);
    float *CoeffsH = static_cast<float *>(sycl::malloc_shared(sizeof(float) * KERNEL_SIZE, q_ct1));
    memcpy(CoeffsH, gaussian_kernel, sizeof(float) * KERNEL_SIZE);

    sobel_kernel_x_gpu =
        (int8_t *)sycl::malloc_device(sizeof(int8_t) * 3 * 3, q_ct1);
    sobel_kernel_y_gpu =
        (int8_t *)sycl::malloc_device(sizeof(int8_t) * 3 * 3, q_ct1);

    q_ct1.memcpy(in_gaussian, ori_pixels,
                 image_height * image_width * sizeof(uint8_t));
    q_ct1.memcpy(sobel_kernel_x_gpu, sobel_kernel_x,
                 sizeof(int8_t) * KERNEL_SIZE * KERNEL_SIZE);
    q_ct1.memcpy(sobel_kernel_y_gpu, sobel_kernel_y,
                 sizeof(int8_t) * KERNEL_SIZE * KERNEL_SIZE)
        .wait();

    constexpr unsigned BLKW = 8;
    constexpr unsigned BLKH = 8;
    auto nWidthInBlk = GET_BLOCK_SIZE(image_width, BLKW);
    auto nHeightInBlk = GET_BLOCK_SIZE(image_height, BLKH);
    auto GlobalRange = cl::sycl::range<2>(nWidthInBlk, nHeightInBlk);
    // Number of workitems in each workgroup.
    cl::sycl::range<2> LocalRange{1, 1};
    cl::sycl::nd_range<2> Range(GlobalRange, LocalRange);
    const unsigned srcWidth = image_width;
    const unsigned srcHeight = image_height;

    /*
    DPCT1012:2: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change
    the way time is measured depending on your goals.
    */
    for (int i = 0; i <= num_iters; i++)
    {
        waitForKernelCompletion();
        /* data transfer image pixels to device */
        // q_ct1.memcpy(in, ori_pixels, image_height * image_width * sizeof(uint8_t));
        // q_ct1.memcpy(gaussian_kernel_gpu, gaussian_kernel,
        //              sizeof(float) * KERNEL_SIZE);

        /* run canny edge detection core - CUDA kernels */
        /* use streams to ensure the kernels are in the same task */
        // sycl::queue *stream;
        // stream = dev_ct1.create_queue();

        // 1. gaussian filter
#ifdef IMAGE_LINUX
        {
#endif
            cl::sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
            cl::sycl::image<2> inputImg(in_gaussian, image_channel_order::r, image_channel_type::unsigned_int8, cl::sycl::range<2>{srcWidth, srcHeight}, props);
            cl::sycl::image<2> outputImg(out_gaussian, image_channel_order::r, image_channel_type::unsigned_int8, cl::sycl::range<2>{srcWidth, srcHeight}, props);
            {
                auto e = q_ct1.submit([&](sycl::handler &cgh)
                                      {
            auto input_acc = inputImg.get_access<cl::sycl::uint4, cl::sycl::access::mode::read>(cgh);
            auto output_acc = outputImg.get_access<cl::sycl::uint4, cl::sycl::access::mode::write>(cgh);

            cgh.parallel_for<class Gaussian>(
                Range, [=](cl::sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {

                using namespace sycl::ext::intel::experimental::esimd;

                uint thread_x = ndi.get_global_id(0);
                uint thread_y = ndi.get_global_id(1);

                uint h_pos = thread_x * BLKW;
                uint v_pos = thread_y * BLKH;

                simd<float, 8> Coeffs = block_load<float, 8>(CoeffsH);
                float Coeffs0 = Coeffs[0];
                float Coeffs1 = Coeffs[1];
                float Coeffs2 = Coeffs[2];

                simd<uchar, (BLKH+2)*BLKW*2> inA_v;
                simd<float, (BLKH+2)*BLKW> mX_v;
                simd<float, BLKW*BLKH> mX_out_v;
                simd<uchar, BLKW*BLKH> outX_v;

                simd<float, (BLKH+2)*BLKW*2> inA_vf;

                auto inA = inA_v.bit_cast_view<uchar, BLKH+2, BLKW*2>();
                auto inAf = inA_vf.bit_cast_view<float, BLKH+2, BLKW*2>();
                auto mX = mX_v.bit_cast_view<float, BLKH+2, BLKW>();
                auto mX_out = mX_out_v.bit_cast_view<float, BLKH, BLKW>();
                auto outX = outX_v.bit_cast_view<uchar, BLKH, BLKW>();

                inA.select<8,1,32,1>(0,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 1, v_pos - 1);
                inA.select<2,1,32,1>(8,0) = media_block_load<uchar, 2, 32>(input_acc, h_pos - 1, v_pos + 7);

                // inA.select<8,1,32,1>(8,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 1, v_pos + 7);
                // inA.select<2,1,32,1>(16,0) = media_block_load<uchar, 2, 32>(input_acc, h_pos - 1, v_pos + 15);

                inA_vf = convert<float, int, BLKW*2*(BLKH+2)>(inA_v);

                mX = (Coeffs0 * inAf.select<BLKH+2,1,BLKW,1>(0,0))
                    + (Coeffs1 * inAf.select<BLKH+2,1,BLKW,1>(0,1))
                    + (Coeffs2 * inAf.select<BLKH+2,1,BLKW,1>(0,2));

                mX_out = (Coeffs0 * mX.select<BLKH,1,BLKW,1>(0,0))
                    + (Coeffs1 * mX.select<BLKH,1,BLKW,1>(1,0))
                    + (Coeffs2 * mX.select<BLKH,1,BLKW,1>(2,0));

                outX_v = saturate<uchar>(mX_out_v + 0.5f);

                media_block_store<uchar, BLKH, BLKW>(output_acc, h_pos, v_pos, outX);
            }); });
                e.wait();

                if (i > 0)
                    elapsed1 += esimd_test::report_time("kernel time", e, e);
            }
#ifdef IMAGE_LINUX
        }
#endif

        stop_ct1 = std::chrono::steady_clock::now(); // start timer
        // q_ct1.
        //     memcpy(out, out_gaussian,
        //             image_height * image_width * sizeof(uint8_t))
        //     .wait();

        waitForKernelCompletion();
        // stop_ct1 = std::chrono::steady_clock::now(); // start timer

        // 2.apply sobel kernels
        /*
        DPCT1049:5: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh)
                     { cgh.parallel_for(
                           sycl::nd_range<3>(sycl::range<3>(1, 1, num_blks) *
                                                 sycl::range<3>(1, 1, thd_per_blk),
                                             sycl::range<3>(1, 1, thd_per_blk)),
                           [=](sycl::nd_item<3> item_ct1)
                           {
                               apply_sobel_filter(gradient_pixels, segment_pixels, out_gaussian,
                                                  image_width, image_height,
                                                  sobel_kernel_x_gpu, sobel_kernel_y_gpu,
                                                  item_ct1);
                           }); })
            .wait();
        q_ct1.memcpy(max_pixels, gradient_pixels,
                     image_height * image_width * sizeof(float))
            .wait();
        waitForKernelCompletion();
        stop_ct2 = std::chrono::steady_clock::now(); // start timer

        // 3. local maxima: non maxima suppression
        /*
        DPCT1049:6: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        q_ct1.submit([&](sycl::handler &cgh)
                     { cgh.parallel_for(
                           sycl::nd_range<3>(sycl::range<3>(1, 1, num_blks) *
                                                 sycl::range<3>(1, 1, thd_per_blk),
                                             sycl::range<3>(1, 1, thd_per_blk)),
                           [=](sycl::nd_item<3> item_ct1)
                           {
                               apply_non_max_suppression(out, out_week, max_pixels, gradient_pixels,
                                                         segment_pixels, image_width,
                                                         image_height, strong_threshold,
                                                         weak_threshold, item_ct1);
                           }); })
            .wait();
        waitForKernelCompletion();
        stop_ct3 = std::chrono::steady_clock::now(); // start timer

        // 4. double threshold
        /*
        DPCT1049:7: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        // q_ct1.submit([&](sycl::handler &cgh){
        //     cgh.parallel_for(
        //     sycl::nd_range<3>(sycl::range<3>(1, 1, num_blks) *
        //                           sycl::range<3>(1, 1, thd_per_blk),
        //                       sycl::range<3>(1, 1, thd_per_blk)),
        //     [=](sycl::nd_item<3> item_ct1) {
        //             apply_double_threshold(out, out_week, max_pixels, strong_threshold,
        //                                    weak_threshold, image_width,
        //                                    image_height, item_ct1);
        //     });
        // }).wait();

        // waitForKernelCompletion();
        // stop_ct4 = std::chrono::steady_clock::now(); // end timer
        // // 5. edges with hysteresis
        // cudaMemcpy(out_week, out, image_height * image_width * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
        // apply_edge_hysteresis <<<num_blks, thd_per_blk, grid, stream >>> (out_week, out, image_width, image_height);

        /*
        DPCT1012:3: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        // stop.wait();
        // waitForKernelCompletion();
        // stop_ct5 = std::chrono::steady_clock::now(); // end timer

        if (i > 0)
        {
            elapsed2 += std::chrono::duration<float, std::milli>(stop_ct2 - stop_ct1)
                            .count();
            elapsed3 += std::chrono::duration<float, std::milli>(stop_ct3 - stop_ct2)
                            .count();
        }
    }

    /* wait for everything to finish */
    // dev_ct1.queues_wait_and_throw();

    /* copy result back to the host */
    q_ct1.memcpy(final_pixels, out, image_width * image_height * sizeof(uint8_t));

    q_ct1.memcpy(week_pixels, out_week,
                 image_width * image_height * sizeof(uint8_t))
        .wait();

    sycl::free(in_gaussian, q_ct1);
    sycl::free(out, q_ct1);
    sycl::free(gradient_pixels, q_ct1);
    sycl::free(max_pixels, q_ct1);
    sycl::free(segment_pixels, q_ct1);
    // sycl::free(gaussian_kernel_gpu, q_ct1);
    sycl::free(sobel_kernel_x_gpu, q_ct1);
    sycl::free(sobel_kernel_y_gpu, q_ct1);
    sycl::free(out_week, q_ct1);

    // elapsed4 = std::chrono::duration<float, std::milli>(stop_ct4 - stop_ct3)
    //   .count();
    /*
    DPCT1026:8: The call to cudaEventDestroy was removed because this call
    is redundant in DPC++.
    */
    /*
    DPCT1026:9: The call to cudaEventDestroy was removed because this call
    is redundant in DPC++.
    */
    // printf("The elapsed time in gpu was %.2f ms\n", elapsed);
    elapsed = elapsed1 + elapsed2 + elapsed3;
    std::cout << "The elapsed time of dpcpp-canny in gpu steps was :" << elapsed / num_iters << " ms" << std::endl;
}

// void apply_gaussian_filter(uint8_t* out_pixels, const uint8_t* in_pixels, int image_width, int image_height, float* gaussian_kernel,
//                            sycl::nd_item<3> item_ct1)
// {
// 	//determine id of thread which corresponds to an individual pixel
//         int pixNum = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
//                      item_ct1.get_local_id(2);
//         const int offset_xy = ((KERNEL_SIZE - 1) / 2);
// 	if (!(pixNum >= 0 && pixNum < image_height * image_width))
// 		return;

// 	int x = pixNum % image_width;
// 	int y = pixNum / image_width;

// 	//Apply Kernel to image
// 	float kernelSum = 0;
// 	float pixelVal = 0;
// 	for (int i = 0; i < KERNEL_SIZE; ++i) {
// 		for (int j = 0; j < KERNEL_SIZE; ++j) {
// 			//check edge cases, if within bounds, apply filter
//             int kernel_x = x + j - offset_xy;
//             int kernel_y = y + i - offset_xy;
//             if ( kernel_x < 0 )
//                 kernel_x = 0;
//             if ( kernel_x >= image_width )
//                 kernel_x = image_width - 1;
//             if ( kernel_y < 0 )
//                 kernel_y = 0;
//             if ( kernel_y >= image_height )
//                 kernel_y = image_height - 1;
//             pixelVal += gaussian_kernel[i * KERNEL_SIZE + j] * in_pixels[kernel_y * image_width + kernel_x];
// 			kernelSum += gaussian_kernel[i * KERNEL_SIZE + j];
// 		}
// 	}
// 	out_pixels[pixNum] = (uint8_t)(pixelVal / kernelSum + float(0.5));

// }

void apply_sobel_filter(float *gradient_pixels, uint8_t *segment_pixels, const uint8_t *in_pixels, int image_width, int image_height, int8_t *sobel_kernel_x, int8_t *sobel_kernel_y,
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
    // int src_pos = x + (y * image_width);

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

    // gradient hypot & direction
    int segment = 0;

    if (convolve_X == 0.0 && convolve_Y == 0.0)
    {
        gradient_pixels[pixNum] = 0;
    }
    else
    {
        gradient_pixels[pixNum] = ((sycl::sqrt(
            (convolve_X * convolve_X) + (convolve_Y * convolve_Y))));
        float theta = sycl::atan2(
            convolve_Y, convolve_X);            // radians. atan2 range: -PI,+PI,
                                                // // theta : 0 - 2PI
        theta = theta * (360.0 / (2.0 * M_PI)); // degrees

        if ((theta <= 22.5 && theta >= -22.5) || (theta <= -157.5) || (theta >= 157.5))
            segment = 1; // "-"
        else if ((theta > 22.5 && theta <= 67.5) || (theta > -157.5 && theta <= -112.5))
            segment = 2; // "/"
        else if ((theta > 67.5 && theta <= 112.5) || (theta >= -112.5 && theta < -67.5))
            segment = 3; // "|"
        else if ((theta >= -67.5 && theta < -22.5) || (theta > 112.5 && theta < 157.5))
            segment = 4; // "\"
    }
    segment_pixels[pixNum] = (uint8_t)segment;
}
void apply_non_max_suppression(uint8_t *out, uint8_t *out_week, float *max_pixels, float *gradient_pixels, uint8_t *segment_pixels, int image_width, int image_height, int strong_threshold, int weak_threshold,
                               sycl::nd_item<3> item_ct1)
{
    int pos = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
              item_ct1.get_local_id(2);
    if (!(pos >= 0 && pos < image_height * image_width))
        return;
    int x = pos % image_width;
    int y = pos / image_width;
    switch (segment_pixels[pos])
    {
    case 1:
        if ((gradient_pixels[pos - 1] >= gradient_pixels[pos] && x > 0) ||
            (gradient_pixels[pos + 1] > gradient_pixels[pos] && x < image_width - 1))
            max_pixels[pos] = 0;
        break;
    case 2:
        if ((gradient_pixels[pos - image_width + 1] >= gradient_pixels[pos] && x < image_width - 1 && y > 0) ||
            (gradient_pixels[pos + image_width - 1] >= gradient_pixels[pos] && y < image_height - 1 && x > 0))
            max_pixels[pos] = 0;
        break;
    case 3:
        if ((gradient_pixels[pos - image_width] >= gradient_pixels[pos] && y > 0) ||
            (gradient_pixels[pos + image_width] > gradient_pixels[pos] && y < image_height - 1))
            max_pixels[pos] = 0;
        break;
    case 4:
        if ((gradient_pixels[pos - image_width - 1] >= gradient_pixels[pos] && x > 0 && y > 0) ||
            (gradient_pixels[pos + (image_width + 1)] >= gradient_pixels[pos] && x < image_width - 1 && y < image_height - 1))
            max_pixels[pos] = 0;
        break;
    default:
        max_pixels[pos] = 0;
        break;
    }

    if (max_pixels[pos] > strong_threshold)
        out[pos] = 255; // absolutely edge
    else if (max_pixels[pos] > weak_threshold)
    {
        out[pos] = 0;
        out_week[pos] = 1; // potential edge
    }
    else
        out[pos] = 0; // absolutely not edge
}
// void apply_double_threshold(uint8_t* out, uint8_t* out_week, float* max_pixels, int strong_threshold, int weak_threshold, int image_width, int image_height,
//                             sycl::nd_item<3> item_ct1) {
//         int pos = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
//                   item_ct1.get_local_id(2);
//         if (!(pos >= 0 && pos < image_height * image_width))
// 		return;
// 	if (max_pixels[pos] > strong_threshold)
// 		out[pos] = 255;      //absolutely edge
// 	else if (max_pixels[pos] > weak_threshold)
//     {
//         out[pos] = 0;
//         out_week[pos] = 1;      //potential edge
//     }
// 	else
// 		out[pos] = 0;       //absolutely not edge
// }
// __global__ void apply_edge_hysteresis(uint8_t* out, uint8_t* in, int image_width, int image_height) {
// 	int pos = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (!(pos >= 0 && pos < image_height * image_width))
// 		return;
// 	if (in[pos] == 100) {
// 		if (in[pos - 1] == 255 || in[pos + 1] == 255 ||
// 			in[pos - image_width] == 255 || in[pos + image_width] == 255 ||
// 			in[pos - image_width - 1] == 255 || in[pos - image_width + 1] == 255 ||
// 			in[pos + image_width - 1] == 255 || in[pos + image_width + 1] == 255)
// 			out[pos] = 255;
// 		else
// 			out[pos] = 0;
// 	}

// }
