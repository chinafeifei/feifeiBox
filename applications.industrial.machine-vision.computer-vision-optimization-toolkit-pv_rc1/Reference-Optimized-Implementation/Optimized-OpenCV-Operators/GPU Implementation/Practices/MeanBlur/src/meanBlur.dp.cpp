//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include "meanBlur.h"
#include "esimd_test_utils.hpp"
// #include <helper_cuda.h>
#include <cmath>
inline int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

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

void meanBlurEsimd3x3(const uint8_t *src, uint8_t *dst, int width, int height, int radio)
{

  sycl::queue q_ct1(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler(), property::queue::enable_profiling{});

  const unsigned srcWidth = width;
  const unsigned srcHeight = height;

  auto dev = q_ct1.get_device();
  auto ctxt = q_ct1.get_context();

  cl::sycl::image<2> inputImg(src, image_channel_order::r, image_channel_type::unsigned_int8, cl::sycl::range<2>{srcWidth, srcHeight});
  cl::sycl::image<2> outputImg(dst, image_channel_order::r, image_channel_type::unsigned_int8, cl::sycl::range<2>{srcWidth, srcHeight});

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

    auto e = q_ct1.submit([&](handler &cgh)
                          {
      auto input_acc = inputImg.get_access<cl::sycl::uint4, cl::sycl::access::mode::read>(cgh);
      auto output_acc = outputImg.get_access<cl::sycl::uint4, cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class meanBlur>(
        Range, [=](cl::sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {

        using namespace sycl::ext::intel::experimental::esimd;

        uint thread_x = ndi.get_global_id(0);
        uint thread_y = ndi.get_global_id(1);

        uint h_pos = thread_x * BLKW;
        uint v_pos = thread_y * BLKH;

        simd<uchar, (BLKH+2)*(BLKW+2)> inA_v;
        simd<float, (BLKH+2)*BLKW> mX_v;
        simd<float, (BLKH+2)*BLKW> mX_v0;
        simd<float, (BLKH+2)*BLKW> mX_v1;
        simd<float, (BLKH+2)*BLKW> mX_v2;

        simd<float, BLKW*BLKH> mX_out_v;
        simd<float, BLKW*BLKH> mX_out_v0;
        simd<float, BLKW*BLKH> mX_out_v1;
        simd<float, BLKW*BLKH> mX_out_v2;
        simd<uchar, BLKW*BLKH> mX_out_v_final;

        auto inA = inA_v.bit_cast_view<uchar, BLKH+2, BLKW+2>();

        auto mX = mX_v.bit_cast_view<float, BLKH+2, BLKW>();
        auto mX_0 = mX_v0.bit_cast_view<float, BLKH+2, BLKW>();
        auto mX_1 = mX_v1.bit_cast_view<float, BLKH+2, BLKW>();
        auto mX_2 = mX_v2.bit_cast_view<float, BLKH+2, BLKW>();

        auto mX_out = mX_out_v.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out0 = mX_out_v0.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out1 = mX_out_v1.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out2 = mX_out_v2.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out_final = mX_out_v_final.bit_cast_view<uchar, BLKH, BLKW>();

        inA.select<8,1,32,1>(0,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 1, v_pos - 1);
        // inA.select<2,1,32,1>(8,0) = media_block_load<uchar, 2, 32>(input_acc, h_pos - 1, v_pos + 7);
        inA.select<8,1,32,1>(8,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 1, v_pos + 7);
        inA.select<2,1,32,1>(16,0) = media_block_load<uchar, 2, 32>(input_acc, h_pos - 1, v_pos + 15);

        using namespace sycl::ext::intel::experimental::esimd;

        mX_0 = inA.select<BLKH+2,1,BLKW,1>(0,0);
        mX_1 = inA.select<BLKH+2,1,BLKW,1>(0,1);
        mX_2 = inA.select<BLKH+2,1,BLKW,1>(0,2);
        
        // mX_v = min(min(mX_v0 , mX_v1), mX_v2);
        mX_v = mX_v0+mX_v1+mX_v2;

        mX_out0 = mX.select<BLKH,1,BLKW,1>(0,0);
        mX_out1 = mX.select<BLKH,1,BLKW,1>(1,0);
        mX_out2 = mX.select<BLKH,1,BLKW,1>(2,0);

        mX_out_v = mX_out_v0 + mX_out_v1 + mX_out_v2;
        mX_out_v_final = saturate<uchar>(mX_out_v/9 + 0.5f);


        media_block_store<uchar, BLKH, BLKW>(output_acc, h_pos, v_pos, mX_out_final);
      }); });
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

  std::cout << "GPU kernel meanBlur time=" << kernel_time << "ms\n";

  return;
}

void meanBlurEsimd5x5(const uint8_t *src, uint8_t *dst, int width, int height, int radio)
{

  sycl::queue q_ct1(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler(), property::queue::enable_profiling{});

  const unsigned srcWidth = width;
  const unsigned srcHeight = height;

  auto dev = q_ct1.get_device();
  auto ctxt = q_ct1.get_context();

  cl::sycl::image<2> inputImg(src, image_channel_order::r, image_channel_type::unsigned_int8, cl::sycl::range<2>{srcWidth, srcHeight});
  cl::sycl::image<2> outputImg(dst, image_channel_order::r, image_channel_type::unsigned_int8, cl::sycl::range<2>{srcWidth, srcHeight});

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

    auto e = q_ct1.submit([&](handler &cgh)
                          {
      auto input_acc = inputImg.get_access<cl::sycl::uint4, cl::sycl::access::mode::read>(cgh);
      auto output_acc = outputImg.get_access<cl::sycl::uint4, cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class meanBlur5x5>(
        Range, [=](cl::sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {

        using namespace sycl::ext::intel::experimental::esimd;

        uint thread_x = ndi.get_global_id(0);
        uint thread_y = ndi.get_global_id(1);

        uint h_pos = thread_x * BLKW;
        uint v_pos = thread_y * BLKH;

        simd<uchar, (BLKH+4)*(BLKW+4)> inA_v;
        simd<float, (BLKH+4)*BLKW> mX_v;
        simd<float, (BLKH+4)*BLKW> mX_v0;
        simd<float, (BLKH+4)*BLKW> mX_v1;
        simd<float, (BLKH+4)*BLKW> mX_v2;
        simd<float, (BLKH+4)*BLKW> mX_v3;
        simd<float, (BLKH+4)*BLKW> mX_v4;

        simd<float, BLKW*BLKH> mX_out_v;
        simd<float, BLKW*BLKH> mX_out_v0;
        simd<float, BLKW*BLKH> mX_out_v1;
        simd<float, BLKW*BLKH> mX_out_v2;
        simd<float, BLKW*BLKH> mX_out_v3;
        simd<float, BLKW*BLKH> mX_out_v4;
        simd<uchar, BLKW*BLKH> mX_out_v_final;

        auto inA = inA_v.bit_cast_view<uchar, BLKH+4, BLKW+4>();

        auto mX = mX_v.bit_cast_view<float, BLKH+4, BLKW>();
        auto mX_0 = mX_v0.bit_cast_view<float, BLKH+4, BLKW>();
        auto mX_1 = mX_v1.bit_cast_view<float, BLKH+4, BLKW>();
        auto mX_2 = mX_v2.bit_cast_view<float, BLKH+4, BLKW>();
        auto mX_3 = mX_v3.bit_cast_view<float, BLKH+4, BLKW>();
        auto mX_4 = mX_v4.bit_cast_view<float, BLKH+4, BLKW>();

        auto mX_out = mX_out_v.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out0 = mX_out_v0.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out1 = mX_out_v1.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out2 = mX_out_v2.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out3 = mX_out_v3.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out4 = mX_out_v4.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out_final = mX_out_v_final.bit_cast_view<uchar, BLKH, BLKW>();


        inA.select<8,1,32,1>(0,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 2, v_pos - 2);
        // inA.select<2,1,32,1>(8,0) = media_block_load<uchar, 2, 32>(input_acc, h_pos - 1, v_pos + 7);
        inA.select<8,1,32,1>(8,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 2, v_pos + 6);
        inA.select<4,1,32,1>(16,0) = media_block_load<uchar, 4, 32>(input_acc, h_pos - 2, v_pos + 14);

        using namespace sycl::ext::intel::experimental::esimd;


        mX_0 = inA.select<BLKH+4,1,BLKW,1>(0,0);
        mX_1 = inA.select<BLKH+4,1,BLKW,1>(0,1);
        mX_2 = inA.select<BLKH+4,1,BLKW,1>(0,2);
        mX_3 = inA.select<BLKH+4,1,BLKW,1>(0,3);
        mX_4 = inA.select<BLKH+4,1,BLKW,1>(0,4);
        
        
        mX_v = mX_v0 +mX_v1+ mX_v2+mX_v3+mX_v4;
        // mX_v = min(min(mX_v , mX_v3), mX_v4);

        mX_out0 = mX.select<BLKH,1,BLKW,1>(0,0);
        mX_out1 = mX.select<BLKH,1,BLKW,1>(1,0);
        mX_out2 = mX.select<BLKH,1,BLKW,1>(2,0);
        mX_out3 = mX.select<BLKH,1,BLKW,1>(3,0);
        mX_out4 = mX.select<BLKH,1,BLKW,1>(4,0);

        mX_out_v = mX_out_v0 +mX_out_v1+ mX_out_v2+mX_out_v3+mX_out_v4;
        mX_out_v_final = saturate<uchar>(mX_out_v/25 + 0.5f);

        media_block_store<uchar, BLKH, BLKW>(output_acc, h_pos, v_pos, mX_out_final);
      }); });
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

  std::cout << "GPU kernel meanBlur time=" << kernel_time << "ms\n";

  return;
}

void meanBlurEsimd11x11(const uint8_t *src, uint8_t *dst, int width, int height, int radio)
{

  sycl::queue q_ct1(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler(), property::queue::enable_profiling{});

  const unsigned srcWidth = width;
  const unsigned srcHeight = height;

  auto dev = q_ct1.get_device();
  auto ctxt = q_ct1.get_context();

  cl::sycl::image<2> inputImg(src, image_channel_order::r, image_channel_type::unsigned_int8, cl::sycl::range<2>{srcWidth, srcHeight});
  cl::sycl::image<2> outputImg(dst, image_channel_order::r, image_channel_type::unsigned_int8, cl::sycl::range<2>{srcWidth, srcHeight});

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

    auto e = q_ct1.submit([&](handler &cgh)
                          {
      auto input_acc = inputImg.get_access<cl::sycl::uint4, cl::sycl::access::mode::read>(cgh);
      auto output_acc = outputImg.get_access<cl::sycl::uint4, cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class meanBlur11x11>(
        Range, [=](cl::sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {

        using namespace sycl::ext::intel::experimental::esimd;

        uint thread_x = ndi.get_global_id(0);
        uint thread_y = ndi.get_global_id(1);

        uint h_pos = thread_x * BLKW;
        uint v_pos = thread_y * BLKH;

        simd<uchar, (BLKH+10)*(BLKW+10)> inA_v;
        simd<float, (BLKH+10)*BLKW> mX_v;
        simd<float, (BLKH+10)*BLKW> mX_v0;
        simd<float, (BLKH+10)*BLKW> mX_v1;
        simd<float, (BLKH+10)*BLKW> mX_v2;
        simd<float, (BLKH+10)*BLKW> mX_v3;
        simd<float, (BLKH+10)*BLKW> mX_v4;
        simd<float, (BLKH+10)*BLKW> mX_v5;
        simd<float, (BLKH+10)*BLKW> mX_v6;
        simd<float, (BLKH+10)*BLKW> mX_v7;
        simd<float, (BLKH+10)*BLKW> mX_v8;
        simd<float, (BLKH+10)*BLKW> mX_v9;
        simd<float, (BLKH+10)*BLKW> mX_v10;

        simd<float, BLKW*BLKH> mX_out_v;
        simd<float, BLKW*BLKH> mX_out_v0;
        simd<float, BLKW*BLKH> mX_out_v1;
        simd<float, BLKW*BLKH> mX_out_v2;
        simd<float, BLKW*BLKH> mX_out_v3;
        simd<float, BLKW*BLKH> mX_out_v4;
        simd<float, BLKW*BLKH> mX_out_v5;
        simd<float, BLKW*BLKH> mX_out_v6;
        simd<float, BLKW*BLKH> mX_out_v7;
        simd<float, BLKW*BLKH> mX_out_v8;
        simd<float, BLKW*BLKH> mX_out_v9;
        simd<float, BLKW*BLKH> mX_out_v10;
        simd<uchar, BLKW*BLKH> mX_out_v_final;

        auto inA = inA_v.bit_cast_view<uchar, BLKH+10, BLKW+10>();

        auto mX = mX_v.bit_cast_view<float, BLKH+10, BLKW>();
        auto mX_0 = mX_v0.bit_cast_view<float, BLKH+10, BLKW>();
        auto mX_1 = mX_v1.bit_cast_view<float, BLKH+10, BLKW>();
        auto mX_2 = mX_v2.bit_cast_view<float, BLKH+10, BLKW>();
        auto mX_3 = mX_v3.bit_cast_view<float, BLKH+10, BLKW>();
        auto mX_4 = mX_v4.bit_cast_view<float, BLKH+10, BLKW>();
        auto mX_5 = mX_v5.bit_cast_view<float, BLKH+10, BLKW>();
        auto mX_6 = mX_v6.bit_cast_view<float, BLKH+10, BLKW>();
        auto mX_7 = mX_v7.bit_cast_view<float, BLKH+10, BLKW>();
        auto mX_8 = mX_v8.bit_cast_view<float, BLKH+10, BLKW>();
        auto mX_9 = mX_v9.bit_cast_view<float, BLKH+10, BLKW>();
        auto mX_10 = mX_v10.bit_cast_view<float, BLKH+10, BLKW>();


        auto mX_out = mX_out_v.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out0 = mX_out_v0.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out1 = mX_out_v1.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out2 = mX_out_v2.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out3 = mX_out_v3.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out4 = mX_out_v4.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out5 = mX_out_v5.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out6 = mX_out_v6.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out7 = mX_out_v7.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out8 = mX_out_v8.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out9 = mX_out_v9.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out10 = mX_out_v10.bit_cast_view<float, BLKH, BLKW>();
        auto mX_out_final = mX_out_v_final.bit_cast_view<uchar, BLKH, BLKW>();


        inA.select<8,1,32,1>(0,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 5, v_pos - 5);
        // inA.select<2,1,32,1>(8,0) = media_block_load<uchar, 2, 32>(input_acc, h_pos - 1, v_pos + 7);
        inA.select<8,1,32,1>(8,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 5, v_pos + 3);
        inA.select<8,1,32,1>(16,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 5, v_pos + 11);
        inA.select<2,1,32,1>(24,0) = media_block_load<uchar, 2, 32>(input_acc, h_pos - 5, v_pos + 19);

        using namespace sycl::ext::intel::experimental::esimd;


        mX_0 = inA.select<BLKH+10,1,BLKW,1>(0,0);
        mX_1 = inA.select<BLKH+10,1,BLKW,1>(0,1);
        mX_2 = inA.select<BLKH+10,1,BLKW,1>(0,2);
        mX_3 = inA.select<BLKH+10,1,BLKW,1>(0,3);
        mX_4 = inA.select<BLKH+10,1,BLKW,1>(0,4);
        mX_5 = inA.select<BLKH+10,1,BLKW,1>(0,5);
        mX_6 = inA.select<BLKH+10,1,BLKW,1>(0,6);
        mX_7 = inA.select<BLKH+10,1,BLKW,1>(0,7);
        mX_8 = inA.select<BLKH+10,1,BLKW,1>(0,8);
        mX_9 = inA.select<BLKH+10,1,BLKW,1>(0,9);
        mX_10 = inA.select<BLKH+10,1,BLKW,1>(0,10);
        
        mX_v = mX_v0 +mX_v1+ mX_v2+mX_v3+mX_v4+mX_v5+ mX_v6+mX_v7+mX_v8+mX_v9+mX_v10;

        mX_out0 = mX.select<BLKH,1,BLKW,1>(0,0);
        mX_out1 = mX.select<BLKH,1,BLKW,1>(1,0);
        mX_out2 = mX.select<BLKH,1,BLKW,1>(2,0);
        mX_out3 = mX.select<BLKH,1,BLKW,1>(3,0);
        mX_out4 = mX.select<BLKH,1,BLKW,1>(4,0);
        mX_out5 = mX.select<BLKH,1,BLKW,1>(5,0);
        mX_out6 = mX.select<BLKH,1,BLKW,1>(6,0);
        mX_out7 = mX.select<BLKH,1,BLKW,1>(7,0);
        mX_out8 = mX.select<BLKH,1,BLKW,1>(8,0);
        mX_out9 = mX.select<BLKH,1,BLKW,1>(9,0);
        mX_out10 = mX.select<BLKH,1,BLKW,1>(10,0);

        mX_out_v = mX_out_v0 +mX_out_v1+ mX_out_v2+mX_out_v3+mX_out_v4+mX_out_v5+ mX_out_v6+mX_out_v7+mX_out_v8+mX_out_v9+mX_out_v10;
        mX_out_v_final = saturate<uchar>(mX_out_v/121 + 0.5f);

        media_block_store<uchar, BLKH, BLKW>(output_acc, h_pos, v_pos, mX_out_final);
      }); });
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

  std::cout << "GPU kernel meanBlur time=" << kernel_time << "ms\n";

  return;
}