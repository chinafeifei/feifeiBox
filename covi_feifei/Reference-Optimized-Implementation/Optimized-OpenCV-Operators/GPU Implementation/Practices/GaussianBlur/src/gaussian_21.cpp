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


inline int esimdGaussian(queue q, cv::Mat src, cv::Mat & dst0, cv::Mat & dst1, cv::Mat & dst2, cv::Mat & dst3, const cv::Size kernelSize, float sigma, float * debug)
{
    const unsigned srcWidth = src.cols;
    const unsigned srcHeight = src.rows;
    /*
    uchar *input = static_cast<uchar *>(malloc_shared(src.cols *  src.rows * sizeof(uchar), dev, ctxt));
    memcpy(input, src.data, src.cols*src.rows*sizeof(uchar));
    */
    dst0 = cv::Mat(srcHeight, srcWidth, CV_32FC1);
    dst1 = cv::Mat(srcHeight, srcWidth, CV_32FC1);
    dst2 = cv::Mat(srcHeight, srcWidth, CV_32FC1);
    dst3 = cv::Mat(srcHeight, srcWidth, CV_32FC1);

    auto dev = q.get_device();
    auto ctxt = q.get_context();

    cl::sycl::image<2> inputImg(src.data, image_channel_order::r, image_channel_type::unsigned_int8, cl::sycl::range<2>{srcWidth, srcHeight});
    cl::sycl::image<2> outputImg0(dst0.data, image_channel_order::r, image_channel_type::fp32, cl::sycl::range<2>{srcWidth, srcHeight});
    cl::sycl::image<2> outputImg1(dst1.data, image_channel_order::r, image_channel_type::fp32, cl::sycl::range<2>{srcWidth, srcHeight});
    cl::sycl::image<2> outputImg2(dst2.data, image_channel_order::r, image_channel_type::fp32, cl::sycl::range<2>{srcWidth, srcHeight});
    cl::sycl::image<2> outputImg3(dst3.data, image_channel_order::r, image_channel_type::fp32, cl::sycl::range<2>{srcWidth, srcHeight});

    cv::Mat coeffs = cv::getGaussianKernel(kernelSize.width, sigma, CV_32F);
    /*
    const sycl::range<1> coeffsSize(coeffs.total() * coeffs.elemSize());
    cl::sycl::buffer<float, 1> coeffsData(coeffs.data, coeffsSize);
    */
    float *CoeffsH = static_cast<float *>(malloc_shared(coeffs.total() * coeffs.elemSize(), dev, ctxt));
    memcpy(CoeffsH, coeffs.data, coeffs.total() * coeffs.elemSize());

    constexpr unsigned BLKW = 8;
    constexpr unsigned BLKH = 8;

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
    
    // kernel 1 f10 * f10
    auto e = q.submit([&](handler &cgh) {
      auto input_acc = inputImg.get_access<cl::sycl::uint4, cl::sycl::access::mode::read>(cgh);
      auto output_acc = outputImg0.get_access<cl::sycl::uint4, cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class Gaussian>(
        Range, [=](cl::sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {

        using namespace sycl::ext::intel::experimental::esimd;

        uint thread_x = ndi.get_global_id(0);
        uint thread_y = ndi.get_global_id(1);

        uint h_pos = thread_x * BLKW;
        uint v_pos = thread_y * BLKH;

        simd<float, 16> Coeffs = block_load<float, 16>(CoeffsH);
        float Coeffs0 = Coeffs[0];
        float Coeffs1 = Coeffs[1];
        float Coeffs2 = Coeffs[2];
        float Coeffs3 = Coeffs[3];
        float Coeffs4 = Coeffs[4];
        float Coeffs5 = Coeffs[5];
        float Coeffs6 = Coeffs[6];
        float Coeffs7 = Coeffs[7];
        float Coeffs8 = Coeffs[8];
        float Coeffs9 = Coeffs[9];
 
        simd<uchar, (BLKH+9)*(BLKW+9)> inA_v;
        simd<float, (BLKH+9)*BLKW> mX_v;
        simd<float, BLKW*BLKH> mX_out_v;
        simd<uchar, BLKW*BLKH> outX_v;

        simd<float, (BLKH+9)*(BLKW+9)> inA_vf;

        auto inA = inA_v.bit_cast_view<uchar, BLKH+9, BLKW+9>();
        auto inAf = inA_vf.bit_cast_view<float, BLKH+9, BLKW+9>();
        auto mX = mX_v.bit_cast_view<float, BLKH+9, BLKW>();
        auto mX_out = mX_out_v.bit_cast_view<float, BLKH, BLKW>();
        // auto outX = outX_v.bit_cast_view<uchar, BLKH, BLKW>();

        inA.select<8,1,32,1>(0,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 10, v_pos - 10);
        inA.select<8,1,32,1>(8,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 10, v_pos - 2);
        inA.select<2,1,32,1>(16,0) = media_block_load<uchar, 2, 32>(input_acc, h_pos - 10, v_pos + 6);


        inA_vf = convert<float, int, (BLKW+9)*(BLKH+9)>(inA_v);

        mX = (Coeffs0 * inAf.select<BLKH+9,1,BLKW,1>(0,0))
            + (Coeffs1 * inAf.select<BLKH+9,1,BLKW,1>(0,1))
            + (Coeffs2 * inAf.select<BLKH+9,1,BLKW,1>(0,2))
            + (Coeffs3 * inAf.select<BLKH+9,1,BLKW,1>(0,3))
            + (Coeffs4 * inAf.select<BLKH+9,1,BLKW,1>(0,4))
            + (Coeffs5 * inAf.select<BLKH+9,1,BLKW,1>(0,5))
            + (Coeffs6 * inAf.select<BLKH+9,1,BLKW,1>(0,6))
            + (Coeffs7 * inAf.select<BLKH+9,1,BLKW,1>(0,7))
            + (Coeffs8 * inAf.select<BLKH+9,1,BLKW,1>(0,8))
            + (Coeffs9 * inAf.select<BLKH+9,1,BLKW,1>(0,9));

        mX_out = (Coeffs0 * mX.select<BLKH,1,BLKW,1>(0,0))
            + (Coeffs1 * mX.select<BLKH,1,BLKW,1>(1,0))
            + (Coeffs2 * mX.select<BLKH,1,BLKW,1>(2,0))
            + (Coeffs3 * mX.select<BLKH,1,BLKW,1>(3,0))
            + (Coeffs4 * mX.select<BLKH,1,BLKW,1>(4,0))
            + (Coeffs5 * mX.select<BLKH,1,BLKW,1>(5,0))
            + (Coeffs6 * mX.select<BLKH,1,BLKW,1>(6,0))
            + (Coeffs7 * mX.select<BLKH,1,BLKW,1>(7,0))
            + (Coeffs8 * mX.select<BLKH,1,BLKW,1>(8,0))
            + (Coeffs9 * mX.select<BLKH,1,BLKW,1>(9,0));

        // outX_v = esimd_sat<uchar>(mX_out_v);

        media_block_store<float, BLKH, BLKW>(output_acc, h_pos * sizeof(float), v_pos, mX_out);
      });
    });
    e.wait();

    // kernel 2 l11 * f10
    auto e1 = q.submit([&](handler &cgh) {
      auto input_acc = inputImg.get_access<cl::sycl::uint4, cl::sycl::access::mode::read>(cgh);
      auto output_acc = outputImg1.get_access<cl::sycl::uint4, cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class Gaussian1>(
        Range, [=](cl::sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {

        using namespace sycl::ext::intel::experimental::esimd;

        uint thread_x = ndi.get_global_id(0);
        uint thread_y = ndi.get_global_id(1);

        uint h_pos = thread_x * BLKW;
        uint v_pos = thread_y * BLKH;

        simd<float, 32> Coeffs = block_load<float, 32>(CoeffsH);
        float Coeffs0 = Coeffs[0];
        float Coeffs1 = Coeffs[1];
        float Coeffs2 = Coeffs[2];
        float Coeffs3 = Coeffs[3];
        float Coeffs4 = Coeffs[4];
        float Coeffs5 = Coeffs[5];
        float Coeffs6 = Coeffs[6];
        float Coeffs7 = Coeffs[7];
        float Coeffs8 = Coeffs[8];
        float Coeffs9 = Coeffs[9];
        float Coeffs10 = Coeffs[10];
        float Coeffs11 = Coeffs[11];
        float Coeffs12 = Coeffs[12];
        float Coeffs13 = Coeffs[13];
        float Coeffs14 = Coeffs[14];
        float Coeffs15 = Coeffs[15];
        float Coeffs16 = Coeffs[16];
        float Coeffs17 = Coeffs[17];
        float Coeffs18 = Coeffs[18];
        float Coeffs19 = Coeffs[19];
        float Coeffs20 = Coeffs[20];
 
        simd<uchar, (BLKH+10)*(BLKW+9)> inA_v;
        simd<float, (BLKH+10)*BLKW> mX_v;
        simd<float, BLKW*BLKH> mX_out_v;
        simd<uchar, BLKW*BLKH> outX_v;

        simd<float, (BLKH+10)*(BLKW+9)> inA_vf;

        auto inA = inA_v.bit_cast_view<uchar, BLKH+10, BLKW+9>();
        auto inAf = inA_vf.bit_cast_view<float, BLKH+10, BLKW+9>();
        auto mX = mX_v.bit_cast_view<float, BLKH+10, BLKW>();
        auto mX_out = mX_out_v.bit_cast_view<float, BLKH, BLKW>();
        // auto outX = outX_v.bit_cast_view<uchar, BLKH, BLKW>();

        inA.select<8,1,32,1>(0,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 10, v_pos);
        inA.select<8,1,32,1>(8,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 10, v_pos + 8);
        inA.select<2,1,32,1>(16,0) = media_block_load<uchar, 2, 32>(input_acc, h_pos - 10, v_pos + 16);
        // inA.select<2,1,32,1>(24,0) = media_block_load<uchar, 2, 32>(input_acc, h_pos - 10, v_pos + 24);


        inA_vf = convert<float, int, (BLKW+9)*(BLKH+10)>(inA_v);

        mX = (Coeffs0 * inAf.select<BLKH+10,1,BLKW,1>(0,0))
            + (Coeffs1 * inAf.select<BLKH+10,1,BLKW,1>(0,1))
            + (Coeffs2 * inAf.select<BLKH+10,1,BLKW,1>(0,2))
            + (Coeffs3 * inAf.select<BLKH+10,1,BLKW,1>(0,3))
            + (Coeffs4 * inAf.select<BLKH+10,1,BLKW,1>(0,4))
            + (Coeffs5 * inAf.select<BLKH+10,1,BLKW,1>(0,5))
            + (Coeffs6 * inAf.select<BLKH+10,1,BLKW,1>(0,6))
            + (Coeffs7 * inAf.select<BLKH+10,1,BLKW,1>(0,7))
            + (Coeffs8 * inAf.select<BLKH+10,1,BLKW,1>(0,8))
            + (Coeffs9 * inAf.select<BLKH+10,1,BLKW,1>(0,9));

        mX_out = (Coeffs10 * mX.select<BLKH,1,BLKW,1>(0,0))
            + (Coeffs11 * mX.select<BLKH,1,BLKW,1>(1,0))
            + (Coeffs12 * mX.select<BLKH,1,BLKW,1>(2,0))
            + (Coeffs13 * mX.select<BLKH,1,BLKW,1>(3,0))
            + (Coeffs14 * mX.select<BLKH,1,BLKW,1>(4,0))
            + (Coeffs15 * mX.select<BLKH,1,BLKW,1>(5,0))
            + (Coeffs16 * mX.select<BLKH,1,BLKW,1>(6,0))
            + (Coeffs17 * mX.select<BLKH,1,BLKW,1>(7,0))
            + (Coeffs18 * mX.select<BLKH,1,BLKW,1>(8,0))
            + (Coeffs19 * mX.select<BLKH,1,BLKW,1>(9,0))
            + (Coeffs20 * mX.select<BLKH,1,BLKW,1>(10,0));

        // outX_v = esimd_sat<uchar>(mX_out_v);

        media_block_store<float, BLKH, BLKW>(output_acc, h_pos * sizeof(float), v_pos, mX_out);
      });
    });
    e1.wait();

    // kernel 3 f10 * l11
    auto e2 = q.submit([&](handler &cgh) {
      auto input_acc = inputImg.get_access<cl::sycl::uint4, cl::sycl::access::mode::read>(cgh);
      auto output_acc = outputImg2.get_access<cl::sycl::uint4, cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class Gaussian2>(
        Range, [=](cl::sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {

        using namespace sycl::ext::intel::experimental::esimd;

        uint thread_x = ndi.get_global_id(0);
        uint thread_y = ndi.get_global_id(1);

        uint h_pos = thread_x * BLKW;
        uint v_pos = thread_y * BLKH;

        simd<float, 32> Coeffs = block_load<float, 32>(CoeffsH);
        float Coeffs0 = Coeffs[0];
        float Coeffs1 = Coeffs[1];
        float Coeffs2 = Coeffs[2];
        float Coeffs3 = Coeffs[3];
        float Coeffs4 = Coeffs[4];
        float Coeffs5 = Coeffs[5];
        float Coeffs6 = Coeffs[6];
        float Coeffs7 = Coeffs[7];
        float Coeffs8 = Coeffs[8];
        float Coeffs9 = Coeffs[9];
        float Coeffs10 = Coeffs[10];
        float Coeffs11 = Coeffs[11];
        float Coeffs12 = Coeffs[12];
        float Coeffs13 = Coeffs[13];
        float Coeffs14 = Coeffs[14];
        float Coeffs15 = Coeffs[15];
        float Coeffs16 = Coeffs[16];
        float Coeffs17 = Coeffs[17];
        float Coeffs18 = Coeffs[18];
        float Coeffs19 = Coeffs[19];
        float Coeffs20 = Coeffs[20];
 
        simd<uchar, (BLKH+9)*(BLKW+10)> inA_v;
        simd<float, (BLKH+9)*BLKW> mX_v;
        simd<float, BLKW*BLKH> mX_out_v;
        simd<uchar, BLKW*BLKH> outX_v;

        simd<float, (BLKH+9)*(BLKW+10)> inA_vf;

        auto inA = inA_v.bit_cast_view<uchar, BLKH+9, BLKW+10>();
        auto inAf = inA_vf.bit_cast_view<float, BLKH+9, BLKW+10>();
        auto mX = mX_v.bit_cast_view<float, BLKH+9, BLKW>();
        auto mX_out = mX_out_v.bit_cast_view<float, BLKH, BLKW>();
        // auto outX = outX_v.bit_cast_view<uchar, BLKH, BLKW>();

        inA.select<8,1,32,1>(0,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos, v_pos - 10);
        inA.select<8,1,32,1>(8,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos, v_pos - 2);
        inA.select<2,1,32,1>(16,0) = media_block_load<uchar, 2, 32>(input_acc, h_pos, v_pos + 6);


        inA_vf = convert<float, int, (BLKW+10)*(BLKH+9)>(inA_v);

        mX = (Coeffs10 * inAf.select<BLKH+9,1,BLKW,1>(0,0))
            + (Coeffs11 * inAf.select<BLKH+9,1,BLKW,1>(0,1))
            + (Coeffs12 * inAf.select<BLKH+9,1,BLKW,1>(0,2))
            + (Coeffs13 * inAf.select<BLKH+9,1,BLKW,1>(0,3))
            + (Coeffs14 * inAf.select<BLKH+9,1,BLKW,1>(0,4))
            + (Coeffs15 * inAf.select<BLKH+9,1,BLKW,1>(0,5))
            + (Coeffs16 * inAf.select<BLKH+9,1,BLKW,1>(0,6))
            + (Coeffs17 * inAf.select<BLKH+9,1,BLKW,1>(0,7))
            + (Coeffs18 * inAf.select<BLKH+9,1,BLKW,1>(0,8))
            + (Coeffs19 * inAf.select<BLKH+9,1,BLKW,1>(0,9))
            + (Coeffs20 * inAf.select<BLKH+9,1,BLKW,1>(0,10));

        mX_out = (Coeffs0 * mX.select<BLKH,1,BLKW,1>(0,0))
            + (Coeffs1 * mX.select<BLKH,1,BLKW,1>(1,0))
            + (Coeffs2 * mX.select<BLKH,1,BLKW,1>(2,0))
            + (Coeffs3 * mX.select<BLKH,1,BLKW,1>(3,0))
            + (Coeffs4 * mX.select<BLKH,1,BLKW,1>(4,0))
            + (Coeffs5 * mX.select<BLKH,1,BLKW,1>(5,0))
            + (Coeffs6 * mX.select<BLKH,1,BLKW,1>(6,0))
            + (Coeffs7 * mX.select<BLKH,1,BLKW,1>(7,0))
            + (Coeffs8 * mX.select<BLKH,1,BLKW,1>(8,0))
            + (Coeffs9 * mX.select<BLKH,1,BLKW,1>(9,0));

        // outX_v = esimd_sat<uchar>(mX_out_v);

        media_block_store<float, BLKH, BLKW>(output_acc, h_pos * sizeof(float), v_pos, mX_out);
      });
    });
    e2.wait();

    // kernel 4 l11 * l11
    auto e3 = q.submit([&](handler &cgh) {
      auto input_acc = inputImg.get_access<cl::sycl::uint4, cl::sycl::access::mode::read>(cgh);
      auto output_acc = outputImg3.get_access<cl::sycl::uint4, cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class Gaussian3>(
        Range, [=](cl::sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {

        using namespace sycl::ext::intel::experimental::esimd;

        uint thread_x = ndi.get_global_id(0);
        uint thread_y = ndi.get_global_id(1);

        uint h_pos = thread_x * BLKW;
        uint v_pos = thread_y * BLKH;

        simd<float, 16> Coeffs = block_load<float, 16>(CoeffsH + 10);
        float Coeffs0 = Coeffs[0];
        float Coeffs1 = Coeffs[1];
        float Coeffs2 = Coeffs[2];
        float Coeffs3 = Coeffs[3];
        float Coeffs4 = Coeffs[4];
        float Coeffs5 = Coeffs[5];
        float Coeffs6 = Coeffs[6];
        float Coeffs7 = Coeffs[7];
        float Coeffs8 = Coeffs[8];
        float Coeffs9 = Coeffs[9];
        float Coeffs10 = Coeffs[10];
 
        simd<uchar, (BLKH+10)*(BLKW+10)> inA_v;
        simd<float, (BLKH+10)*BLKW> mX_v;
        simd<float, BLKW*BLKH> mX_out_v;
        simd<uchar, BLKW*BLKH> outX_v;

        simd<float, (BLKH+10)*(BLKW+10)> inA_vf;

        auto inA = inA_v.bit_cast_view<uchar, BLKH+10, BLKW+10>();
        auto inAf = inA_vf.bit_cast_view<float, BLKH+10, BLKW+10>();
        auto mX = mX_v.bit_cast_view<float, BLKH+10, BLKW>();
        auto mX_out = mX_out_v.bit_cast_view<float, BLKH, BLKW>();
        // auto outX = outX_v.bit_cast_view<uchar, BLKH, BLKW>();

        inA.select<8,1,32,1>(0,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos, v_pos);
        inA.select<8,1,32,1>(8,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos, v_pos + 8);
        inA.select<2,1,32,1>(16,0) = media_block_load<uchar, 2, 32>(input_acc, h_pos, v_pos + 16);


        inA_vf = convert<float, int, (BLKW+10)*(BLKH+10)>(inA_v);

        mX = (Coeffs0 * inAf.select<BLKH+10,1,BLKW,1>(0,0))
            + (Coeffs1 * inAf.select<BLKH+10,1,BLKW,1>(0,1))
            + (Coeffs2 * inAf.select<BLKH+10,1,BLKW,1>(0,2))
            + (Coeffs3 * inAf.select<BLKH+10,1,BLKW,1>(0,3))
            + (Coeffs4 * inAf.select<BLKH+10,1,BLKW,1>(0,4))
            + (Coeffs5 * inAf.select<BLKH+10,1,BLKW,1>(0,5))
            + (Coeffs6 * inAf.select<BLKH+10,1,BLKW,1>(0,6))
            + (Coeffs7 * inAf.select<BLKH+10,1,BLKW,1>(0,7))
            + (Coeffs8 * inAf.select<BLKH+10,1,BLKW,1>(0,8))
            + (Coeffs9 * inAf.select<BLKH+10,1,BLKW,1>(0,9))
            + (Coeffs10 * inAf.select<BLKH+10,1,BLKW,1>(0,10));

        mX_out = (Coeffs0 * mX.select<BLKH,1,BLKW,1>(0,0))
            + (Coeffs1 * mX.select<BLKH,1,BLKW,1>(1,0))
            + (Coeffs2 * mX.select<BLKH,1,BLKW,1>(2,0))
            + (Coeffs3 * mX.select<BLKH,1,BLKW,1>(3,0))
            + (Coeffs4 * mX.select<BLKH,1,BLKW,1>(4,0))
            + (Coeffs5 * mX.select<BLKH,1,BLKW,1>(5,0))
            + (Coeffs6 * mX.select<BLKH,1,BLKW,1>(6,0))
            + (Coeffs7 * mX.select<BLKH,1,BLKW,1>(7,0))
            + (Coeffs8 * mX.select<BLKH,1,BLKW,1>(8,0))
            + (Coeffs9 * mX.select<BLKH,1,BLKW,1>(9,0))
            + (Coeffs10 * mX.select<BLKH,1,BLKW,1>(10,0));

        // outX_v = esimd_sat<uchar>(mX_out_v);

        media_block_store<float, BLKH, BLKW>(output_acc, h_pos * sizeof(float), v_pos, mX_out);
      });
    });
    e3.wait();

    etime = esimd_test::report_time("kernel time", e, e);
    etime += esimd_test::report_time("kernel time", e1, e1);
    etime += esimd_test::report_time("kernel time", e2, e2);
    etime += esimd_test::report_time("kernel time", e3, e3);

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
  
  const float sigma = 4.3f;
  int kernel_size = 21;

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

  cv::Mat esimdDst, esimdDst0, esimdDst1, esimdDst2, esimdDst3;
  esimdGaussian(q, src, esimdDst0, esimdDst1, esimdDst2, esimdDst3, cv::Size(kernel_size, kernel_size), sigma, debug);

  esimdDst = esimdDst0 + esimdDst1 + esimdDst2 + esimdDst3;
  esimdDst.convertTo(esimdDst, CV_8U);

  /*
  cv::imshow("cvoutput", cvdst);
  cv::imshow("esimdoutput", esimdDst);
  cv::waitKey(0);
  */
  cv::cvtColor(cvdst, cvdst, cv::COLOR_GRAY2BGR);
  cv::cvtColor(esimdDst, esimdDst, cv::COLOR_GRAY2BGR);
  cv::imwrite("./data/gaussian_21.ppm", cvdst);
  cv::imwrite("./data/gaussian_21_dpcpp.ppm", esimdDst);

  for (int j = 0; j < cvdst.rows; j++)
    for (int i = 0; i < cvdst.cols; i++)
        if (abs( cvdst.data[j*cvdst.cols + i] - esimdDst.data[j*cvdst.cols + i]) > 1)
        {
            std::cout << "mismatch at i=" << i << " j=" << j << std::endl;
            printf("cvdata = %d, esimd = %d\n", cvdst.data[j*cvdst.cols+i],
                    esimdDst.data[j*cvdst.cols+i]);
        }

  return 0;
}
