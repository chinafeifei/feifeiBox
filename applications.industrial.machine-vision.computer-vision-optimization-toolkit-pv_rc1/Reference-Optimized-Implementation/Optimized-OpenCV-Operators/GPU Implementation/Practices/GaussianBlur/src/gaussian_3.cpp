//==---------------- vadd_usm.cpp  - DPC++ ESIMD on-device test ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <iostream>
#include <assert.h>
#include <chrono>
// #include <opencv2/opencv.hpp>
#include "esimd_test_utils.hpp"

#define GET_BLOCK_SIZE(t, blk) (t + blk - 1)/blk

using namespace cl::sycl;


inline int esimdGaussian(queue q, cv::Mat src, cv::Mat & dst, const cv::Size kernelSize, float sigma, float * debug)
{
    const unsigned srcWidth = src.cols;
    const unsigned srcHeight = src.rows;
    /*
    uchar *input = static_cast<uchar *>(malloc_shared(src.cols *  src.rows * sizeof(uchar), dev, ctxt));
    memcpy(input, src.data, src.cols*src.rows*sizeof(uchar));
    */
    dst = cv::Mat(srcHeight, srcWidth, CV_8UC1);

    auto dev = q.get_device();
    auto ctxt = q.get_context();

    cl::sycl::image<2> inputImg(src.data, image_channel_order::r, image_channel_type::unsigned_int8, cl::sycl::range<2>{srcWidth, srcHeight});
    cl::sycl::image<2> outputImg(dst.data, image_channel_order::r, image_channel_type::unsigned_int8, cl::sycl::range<2>{srcWidth, srcHeight});

    cv::Mat coeffs = cv::getGaussianKernel(kernelSize.width, sigma, CV_32F);
    /*
    const sycl::range<1> coeffsSize(coeffs.total() * coeffs.elemSize());
    cl::sycl::buffer<float, 1> coeffsData(coeffs.data, coeffsSize);
    */
    float *CoeffsH = static_cast<float *>(malloc_shared(coeffs.total() * coeffs.elemSize(), dev, ctxt));
    memcpy(CoeffsH, coeffs.data, coeffs.total() * coeffs.elemSize());

    constexpr unsigned BLKW = 16;
    constexpr unsigned BLKH = 16;

    auto nWidthInBlk = GET_BLOCK_SIZE(srcWidth, BLKW);
    auto nHeightInBlk = GET_BLOCK_SIZE(srcHeight, BLKH);

    auto GlobalRange = cl::sycl::range<2>(nWidthInBlk, nHeightInBlk);
    // Number of workitems in each workgroup.
    cl::sycl::range<2> LocalRange{1, 1};
    cl::sycl::nd_range<2> Range(GlobalRange, LocalRange);

    double etime = 0;
    double start;
    double kernel_times = 0;
    float iters = 5;

    esimd_test::Timer timer;

    for (int i = 0; i <= iters; i++)
    {

    auto e = q.submit([&](handler &cgh) {
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
        inA.select<8,1,32,1>(8,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 1, v_pos + 7);
        inA.select<2,1,32,1>(16,0) = media_block_load<uchar, 2, 32>(input_acc, h_pos - 1, v_pos + 15);

        inA_vf = convert<float, int, BLKW*2*(BLKH+2)>(inA_v);

        mX = (Coeffs0 * inAf.select<BLKH+2,1,BLKW,1>(0,0))
            + (Coeffs1 * inAf.select<BLKH+2,1,BLKW,1>(0,1))
            + (Coeffs2 * inAf.select<BLKH+2,1,BLKW,1>(0,2));

        mX_out = (Coeffs0 * mX.select<BLKH,1,BLKW,1>(0,0))
            + (Coeffs1 * mX.select<BLKH,1,BLKW,1>(1,0))
            + (Coeffs2 * mX.select<BLKH,1,BLKW,1>(2,0));

        outX_v = saturate<uchar>(mX_out_v + 0.5f);

        media_block_store<uchar, BLKH, BLKW>(output_acc, h_pos, v_pos, outX);
      });
    });
    e.wait();
    etime = esimd_test::report_time("kernel time", e, e);

    if (i > 0)
        kernel_times += etime;
    else
        start = timer.Elapsed();
    }

   double end = timer.Elapsed();

   float total_time = (end - start) * 1000.f / iters;
   float kernel_time = kernel_times / iters;

    std::cout << "GPU kernel time=" << kernel_time << " ms\n";
    // std::cout << "GPU total time=" << total_time << " ms\n";
      /*
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return e.get_cl_code();
    }
    */
    //double end = timer.Elapsed();
    //float total_time = (end - start) * 1000.0f;

    //printf("one time = %f\n", total_time);

    return 0;
}



int main(void) {

  const float sigma = 1.f;
  int kernel_size = 3;
  
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler(), property::queue::enable_profiling{});

  auto dev = q.get_device();
  // std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  std::cout << "gaussian of (" << kernel_size << "*" << kernel_size << "):" << std::endl;
  auto ctxt = q.get_context();

  cv::Mat src = cv::imread("./data/color_4288.jpg", cv::IMREAD_GRAYSCALE);
  float *debug =
      static_cast<float *>(malloc_shared(256 * sizeof(float), dev, ctxt));

  memset(debug , 0, 256*sizeof(float));

  cv::Mat cvdst;

  esimd_test::Timer timer;
  unsigned num_iters = 5;
  double start = timer.Elapsed();

  for (int i = 0; i < num_iters; i++)
    cv::GaussianBlur(src, cvdst, cv::Size(kernel_size, kernel_size), sigma, sigma, cv::BORDER_REPLICATE);

  double end = timer.Elapsed();
  float total_time = (end - start) * 1000.0f/num_iters;
  printf("CPU OpenCV time = %f ms\n", total_time);

  cv::Mat esimdDst;
  esimdGaussian(q, src, esimdDst, cv::Size(kernel_size, kernel_size), sigma, debug);

  /*
  cv::imshow("cvoutput", cvdst);
  cv::imshow("esimdoutput", esimdDst);
  cv::waitKey(0);
  */
  cv::cvtColor(cvdst, cvdst, cv::COLOR_GRAY2BGR);
  cv::cvtColor(esimdDst, esimdDst, cv::COLOR_GRAY2BGR);
  cv::imwrite("./data/gaussian_3.ppm", cvdst);
  cv::imwrite("./data/gaussian_3_dpcpp.ppm", esimdDst);

  for (int j = 0; j < cvdst.rows; j++)
    for (int i = 0; i < cvdst.cols; i++)
        if (abs( cvdst.data[j*cvdst.cols + i] - esimdDst.data[j*cvdst.cols + i]) > 1)
        {
            std::cout << "mismatch at i=" << i << " j=" << j << std::endl;
            printf("cvdata = %d, esimd = %d\n", cvdst.data[j*cvdst.cols+i],
                    esimdDst.data[j*cvdst.cols+i]);
        }

  std::cout << std::endl;
  return 0;
}
