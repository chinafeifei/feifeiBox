//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include "MorphOpen.h"
#include "esimd_test_utils.hpp"
// #include <helper_cuda.h>
#include <cmath>
// helper for CUDA Error handling and initialization

/**
 * Naive erosion kernel with each thread processing a square area.
 */
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

void MorphOpenEsimd(const uint8_t *src, uint8_t *dst_tmp, uint8_t *dst, int width, int height, int radio)
{

  sycl::queue q_ct1(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler(), property::queue::enable_profiling{});

  const unsigned srcWidth = width;
  const unsigned srcHeight = height;

  auto dev = q_ct1.get_device();
  auto ctxt = q_ct1.get_context();

  cl::sycl::image<2> inputImg(src, image_channel_order::r, image_channel_type::unsigned_int8, cl::sycl::range<2>{srcWidth, srcHeight});
  cl::sycl::image<2> tmpImg(dst_tmp, image_channel_order::r, image_channel_type::unsigned_int8, cl::sycl::range<2>{srcWidth, srcHeight});
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
      auto output_acc = tmpImg.get_access<cl::sycl::uint4, cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class Erosion>(
        Range, [=](cl::sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {

        using namespace sycl::ext::intel::experimental::esimd;

        uint thread_x = ndi.get_global_id(0);
        uint thread_y = ndi.get_global_id(1);

        uint h_pos = thread_x * BLKW;
        uint v_pos = thread_y * BLKH;

        simd<uchar, (BLKH+2)*(BLKW+2)> inA_v;
        simd<uchar, (BLKH+2)*BLKW> mX_v;
        simd<uchar, (BLKH+2)*BLKW> mX_v0;
        simd<uchar, (BLKH+2)*BLKW> mX_v1;
        simd<uchar, (BLKH+2)*BLKW> mX_v2;

        simd<uchar, BLKW*BLKH> mX_out_v;
        simd<uchar, BLKW*BLKH> mX_out_v0;
        simd<uchar, BLKW*BLKH> mX_out_v1;
        simd<uchar, BLKW*BLKH> mX_out_v2;

        auto inA = inA_v.bit_cast_view<uchar, BLKH+2, BLKW+2>();

        auto mX = mX_v.bit_cast_view<uchar, BLKH+2, BLKW>();
        auto mX_0 = mX_v0.bit_cast_view<uchar, BLKH+2, BLKW>();
        auto mX_1 = mX_v1.bit_cast_view<uchar, BLKH+2, BLKW>();
        auto mX_2 = mX_v2.bit_cast_view<uchar, BLKH+2, BLKW>();

        auto mX_out = mX_out_v.bit_cast_view<uchar, BLKH, BLKW>();
        auto mX_out0 = mX_out_v0.bit_cast_view<uchar, BLKH, BLKW>();
        auto mX_out1 = mX_out_v1.bit_cast_view<uchar, BLKH, BLKW>();
        auto mX_out2 = mX_out_v2.bit_cast_view<uchar, BLKH, BLKW>();

        inA.select<8,1,32,1>(0,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 1, v_pos - 1);
        // inA.select<2,1,32,1>(8,0) = media_block_load<uchar, 2, 32>(input_acc, h_pos - 1, v_pos + 7);
        inA.select<8,1,32,1>(8,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 1, v_pos + 7);
        inA.select<2,1,32,1>(16,0) = media_block_load<uchar, 2, 32>(input_acc, h_pos - 1, v_pos + 15);

        using namespace sycl::ext::intel::experimental::esimd;

        mX_0 = inA.select<BLKH+2,1,BLKW,1>(0,0);
        mX_1 = inA.select<BLKH+2,1,BLKW,1>(0,1);
        mX_2 = inA.select<BLKH+2,1,BLKW,1>(0,2);
        
        mX_v = min(min(mX_v0 , mX_v1), mX_v2);

        mX_out0 = mX.select<BLKH,1,BLKW,1>(0,0);
        mX_out1 = mX.select<BLKH,1,BLKW,1>(1,0);
        mX_out2 = mX.select<BLKH,1,BLKW,1>(2,0);

        mX_out_v = min(min(mX_out_v0 , mX_out_v1), mX_out_v2);

        media_block_store<uchar, BLKH, BLKW>(output_acc, h_pos, v_pos, mX_out);
      }); });
    e.wait();

    auto e1 = q_ct1.submit([&](handler &cgh)
                           {
      auto input_acc = tmpImg.get_access<cl::sycl::uint4, cl::sycl::access::mode::read>(cgh);
      auto output_acc = outputImg.get_access<cl::sycl::uint4, cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class Dilation>(
        Range, [=](cl::sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {

        using namespace sycl::ext::intel::experimental::esimd;

        uint thread_x = ndi.get_global_id(0);
        uint thread_y = ndi.get_global_id(1);

        uint h_pos = thread_x * BLKW;
        uint v_pos = thread_y * BLKH;

        simd<uchar, (BLKH+2)*(BLKW+2)> inA_v;
        simd<uchar, (BLKH+2)*BLKW> mX_v;
        simd<uchar, (BLKH+2)*BLKW> mX_v0;
        simd<uchar, (BLKH+2)*BLKW> mX_v1;
        simd<uchar, (BLKH+2)*BLKW> mX_v2;

        simd<uchar, BLKW*BLKH> mX_out_v;
        simd<uchar, BLKW*BLKH> mX_out_v0;
        simd<uchar, BLKW*BLKH> mX_out_v1;
        simd<uchar, BLKW*BLKH> mX_out_v2;

        auto inA = inA_v.bit_cast_view<uchar, BLKH+2, BLKW+2>();

        auto mX = mX_v.bit_cast_view<uchar, BLKH+2, BLKW>();
        auto mX_0 = mX_v0.bit_cast_view<uchar, BLKH+2, BLKW>();
        auto mX_1 = mX_v1.bit_cast_view<uchar, BLKH+2, BLKW>();
        auto mX_2 = mX_v2.bit_cast_view<uchar, BLKH+2, BLKW>();

        auto mX_out = mX_out_v.bit_cast_view<uchar, BLKH, BLKW>();
        auto mX_out0 = mX_out_v0.bit_cast_view<uchar, BLKH, BLKW>();
        auto mX_out1 = mX_out_v1.bit_cast_view<uchar, BLKH, BLKW>();
        auto mX_out2 = mX_out_v2.bit_cast_view<uchar, BLKH, BLKW>();

        inA.select<8,1,32,1>(0,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 1, v_pos - 1);
        inA.select<8,1,32,1>(8,0) = media_block_load<uchar, 8, 32>(input_acc, h_pos - 1, v_pos + 7);
        inA.select<2,1,32,1>(16,0) = media_block_load<uchar, 2, 32>(input_acc, h_pos - 1, v_pos + 15);

        using namespace sycl::ext::intel::experimental::esimd;

        mX_0 = inA.select<BLKH+2,1,BLKW,1>(0,0);
        mX_1 = inA.select<BLKH+2,1,BLKW,1>(0,1);
        mX_2 = inA.select<BLKH+2,1,BLKW,1>(0,2);
        
        mX_v = max(max(mX_v0 , mX_v1), mX_v2);

        mX_out0 = mX.select<BLKH,1,BLKW,1>(0,0);
        mX_out1 = mX.select<BLKH,1,BLKW,1>(1,0);
        mX_out2 = mX.select<BLKH,1,BLKW,1>(2,0);

        mX_out_v = max(max(mX_out_v0 , mX_out_v1), mX_out_v2);

        media_block_store<uchar, BLKH, BLKW>(output_acc, h_pos, v_pos, mX_out);
      }); });
    e1.wait();

    etime = esimd_test::report_time("kernel time", e, e);
    etime += esimd_test::report_time("kernel time", e1, e1);

    if (i > 0)
      kernel_times += etime;
    else
      start = timer.Elapsed();
  }

  double end = timer.Elapsed();

  float total_time = (end - start) * 1000.f / iters;
  float kernel_time = kernel_times / iters;

  std::cout << "GPU kernel MorphOpen time=" << kernel_time << " ms\n";
  // std::cout << "GPU total time=" << total_time << " ms\n";
  /*
} catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return e.get_cl_code();
}
*/
  // double end = timer.Elapsed();
  // float total_time = (end - start) * 1000.0f;

  // printf("one time = %f\n", total_time);

  return;
}