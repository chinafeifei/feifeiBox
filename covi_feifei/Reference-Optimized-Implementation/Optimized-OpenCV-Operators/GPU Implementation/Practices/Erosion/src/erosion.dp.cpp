//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include "erosion.h"
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

void NaiveErosionKernel(int *src, int *dst, int width, int height, int radio,
                        sycl::nd_item<3> item_ct1)
{
    int x = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
    int y = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
            item_ct1.get_local_id(1);
    if (y >= height || x >= width)
    {
        return;
    }
    unsigned int start_i = sycl::max((int)(y - radio), 0);
    unsigned int end_i = sycl::min((int)(height - 1), (int)(y + radio));
    unsigned int start_j = sycl::max((int)(x - radio), 0);
    unsigned int end_j = sycl::min((int)(width - 1), (int)(x + radio));
    int value = 255;
    for (int i = start_i; i <= end_i; i++)
    {
        for (int j = start_j; j <= end_j; j++)
        {
            value = sycl::min(value, src[i * width + j]);
        }
    }
    dst[y * width + x] = value;
}

void NaiveErosion(int *src, int *dst, int width, int height, int radio)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    sycl::range<3> block(1, 16, 16);
    sycl::range<3> grid(1, iDivUp(height, block[1]),
                        iDivUp(width, block[2]));
    /*
    DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    q_ct1.submit([&](sycl::handler &cgh)
                 { cgh.parallel_for(
                       sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1)
                       { NaiveErosionKernel(src, dst, width, height, radio, item_ct1); }); });
    /*
    DPCT1003:1: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    dpct::get_current_device().queues_wait_and_throw();
}

/**
 * Two steps erosion using separable filters
 */
void ErosionStep2(int *src, int *dst, int width, int height, int radio,
                  sycl::nd_item<3> item_ct1)
{
    int x = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
    int y = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
            item_ct1.get_local_id(1);
    if (y >= height || x >= width)
    {
        return;
    }
    unsigned int start_i = sycl::max((int)(y - radio), 0);
    unsigned int end_i = sycl::min((int)(height - 1), (int)(y + radio));
    int value = 255;
    for (int i = start_i; i <= end_i; i++)
    {
        value = sycl::min(value, src[i * width + x]);
    }
    dst[y * width + x] = value;
}

void ErosionStep1(int *src, int *dst, int width, int height, int radio,
                  sycl::nd_item<3> item_ct1)
{
    int x = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
    int y = item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
            item_ct1.get_local_id(1);
    if (y >= height || x >= width)
    {
        return;
    }
    unsigned int start_j = sycl::max((int)(x - radio), 0);
    unsigned int end_j = sycl::min((int)(width - 1), (int)(x + radio));
    int value = 255;
    for (int j = start_j; j <= end_j; j++)
    {
        value = sycl::min(value, src[y * width + j]);
    }
    dst[y * width + x] = value;
}

void ErosionTwoSteps(int *src, int *dst, int *temp, int width, int height,
                     int radio)
try
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    sycl::range<3> block(1, 16, 16);
    sycl::range<3> grid(1, iDivUp(height, block[1]),
                        iDivUp(width, block[2]));
    /*
    DPCT1049:2: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    q_ct1.submit([&](sycl::handler &cgh)
                 { cgh.parallel_for(
                       sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1)
                       { ErosionStep1(src, temp, width, height, radio, item_ct1); }); });
    /*
    DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    dpct::get_current_device().queues_wait_and_throw();
    /*
    DPCT1049:3: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    q_ct1.submit([&](sycl::handler &cgh)
                 { cgh.parallel_for(
                       sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1)
                       { ErosionStep2(temp, dst, width, height, radio, item_ct1); }); });
    /*
    DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    dpct::get_current_device().queues_wait_and_throw();
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

/**
 * Two steps erosion using separable filters with shared memory.
 */
void ErosionSharedStep2(int *src, int *src_src, int *dst, int radio, int width, int height, int tile_w, int tile_h,
                        sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto smem = (int *)dpct_local;
    int tx = item_ct1.get_local_id(2);
    int ty = item_ct1.get_local_id(1);
    int bx = item_ct1.get_group(2);
    int by = item_ct1.get_group(1);
    int x = bx * tile_w + tx;
    int y = by * tile_h + ty - radio;
    smem[ty * item_ct1.get_local_range().get(2) + tx] = 255;
    /*
    DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (x >= width || y < 0 || y >= height)
    {
        return;
    }
    smem[ty * item_ct1.get_local_range().get(2) + tx] = src[y * width + x];
    /*
    DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (y < (by * tile_h) || y >= ((by + 1) * tile_h))
    {
        return;
    }
    int *smem_thread =
        &smem[(ty - radio) * item_ct1.get_local_range().get(2) + tx];
    int val = smem_thread[0];
    for (int yy = 1; yy <= 2 * radio; yy++)
    {
        val = sycl::min(val, smem_thread[yy * item_ct1.get_local_range(2)]);
    }
    dst[y * width + x] = val;
}

void ErosionSharedStep1(int *src, int *dst, int radio, int width, int height, int tile_w, int tile_h,
                        sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto smem = (int *)dpct_local;
    int tx = item_ct1.get_local_id(2);
    int ty = item_ct1.get_local_id(1);
    int bx = item_ct1.get_group(2);
    int by = item_ct1.get_group(1);
    int x = bx * tile_w + tx - radio;
    int y = by * tile_h + ty;
    smem[ty * item_ct1.get_local_range().get(2) + tx] = 255;
    /*
    DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (x < 0 || x >= width || y >= height)
    {
        return;
    }
    smem[ty * item_ct1.get_local_range().get(2) + tx] = src[y * width + x];
    /*
    DPCT1065:9: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (x < (bx * tile_w) || x >= ((bx + 1) * tile_w))
    {
        return;
    }
    int *smem_thread =
        &smem[ty * item_ct1.get_local_range().get(2) + tx - radio];
    int val = smem_thread[0];
    for (int xx = 1; xx <= 2 * radio; xx++)
    {
        val = sycl::min(val, smem_thread[xx]);
    }
    dst[y * width + x] = val;
    item_ct1.barrier(sycl::access::fence_space::local_space);
}

void ErosionTwoStepsShared(int *src, int *dst, int *temp, int width, int height,
                           int radio)
try
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    int tile_w = 320;
    int tile_h = 1;
    sycl::range<3> block2(1, tile_h, tile_w + (2 * radio));
    sycl::range<3> grid2(1, iDivUp(height, tile_h),
                         iDivUp(width, tile_w));
    /*
    DPCT1049:10: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    q_ct1.submit([&](sycl::handler &cgh)
                 {
        /*
        DPCT1083:26: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(
                sycl::range<1>(block2[1] * block2[2] * sizeof(int)), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid2 * block2, block2),
                         [=](sycl::nd_item<3> item_ct1) {
                             ErosionSharedStep1(
                                 src, temp, radio, width, height, tile_w,
                                 tile_h, item_ct1,
                                 dpct_local_acc_ct1.get_pointer());
                         }); });
    /*
    DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    dpct::get_current_device().queues_wait_and_throw();
    tile_w = 8;
    tile_h = 32;
    sycl::range<3> block3(1, tile_h + (2 * radio), tile_w);
    sycl::range<3> grid3(1, iDivUp(height, tile_h),
                         iDivUp(width, tile_w));
    /*
    DPCT1049:11: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    q_ct1.submit([&](sycl::handler &cgh)
                 {
        /*
        DPCT1083:27: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(
                sycl::range<1>(block3[1] * block3[2] * sizeof(int)), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid3 * block3, block3),
                         [=](sycl::nd_item<3> item_ct1) {
                             ErosionSharedStep2(
                                 temp, src, dst, radio, width, height, tile_w,
                                 tile_h, item_ct1,
                                 dpct_local_acc_ct1.get_pointer());
                         }); });
    /*
    DPCT1003:13: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    dpct::get_current_device().queues_wait_and_throw();
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void ErosionEsimd(const uint8_t *src, uint8_t *dst, int width, int height, int radio)
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

        etime = esimd_test::report_time("kernel time", e, e);

        if (i > 0)
            kernel_times += etime;
        else
            start = timer.Elapsed();
    }

    double end = timer.Elapsed();

    float total_time = (end - start) * 1000.f / iters;
    float kernel_time = kernel_times / iters;

    std::cout << "GPU kernel erosion time=" << kernel_time << "ms\n";

    return;
}

void ErosionEsimd5x5(const uint8_t *src, uint8_t *dst, int width, int height, int radio)
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

      cgh.parallel_for<class Erosion5x5>(
        Range, [=](cl::sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {

        using namespace sycl::ext::intel::experimental::esimd;

        uint thread_x = ndi.get_global_id(0);
        uint thread_y = ndi.get_global_id(1);

        uint h_pos = thread_x * BLKW;
        uint v_pos = thread_y * BLKH;

        simd<uchar, (BLKH+4)*(BLKW+4)> inA_v;
        simd<uchar, (BLKH+4)*BLKW> mX_v;
        simd<uchar, (BLKH+4)*BLKW> mX_v0;
        simd<uchar, (BLKH+4)*BLKW> mX_v1;
        simd<uchar, (BLKH+4)*BLKW> mX_v2;
        simd<uchar, (BLKH+4)*BLKW> mX_v3;
        simd<uchar, (BLKH+4)*BLKW> mX_v4;

        simd<uchar, BLKW*BLKH> mX_out_v;
        simd<uchar, BLKW*BLKH> mX_out_v0;
        simd<uchar, BLKW*BLKH> mX_out_v1;
        simd<uchar, BLKW*BLKH> mX_out_v2;
        simd<uchar, BLKW*BLKH> mX_out_v3;
        simd<uchar, BLKW*BLKH> mX_out_v4;

        auto inA = inA_v.bit_cast_view<uchar, BLKH+4, BLKW+4>();

        auto mX = mX_v.bit_cast_view<uchar, BLKH+4, BLKW>();
        auto mX_0 = mX_v0.bit_cast_view<uchar, BLKH+4, BLKW>();
        auto mX_1 = mX_v1.bit_cast_view<uchar, BLKH+4, BLKW>();
        auto mX_2 = mX_v2.bit_cast_view<uchar, BLKH+4, BLKW>();
        auto mX_3 = mX_v3.bit_cast_view<uchar, BLKH+4, BLKW>();
        auto mX_4 = mX_v4.bit_cast_view<uchar, BLKH+4, BLKW>();

        auto mX_out = mX_out_v.bit_cast_view<uchar, BLKH, BLKW>();
        auto mX_out0 = mX_out_v0.bit_cast_view<uchar, BLKH, BLKW>();
        auto mX_out1 = mX_out_v1.bit_cast_view<uchar, BLKH, BLKW>();
        auto mX_out2 = mX_out_v2.bit_cast_view<uchar, BLKH, BLKW>();
        auto mX_out3 = mX_out_v3.bit_cast_view<uchar, BLKH, BLKW>();
        auto mX_out4 = mX_out_v4.bit_cast_view<uchar, BLKH, BLKW>();


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
        
        
        mX_v = min(min(min(min(mX_v0 , mX_v1), mX_v2),mX_v3),mX_v4);
        // mX_v = min(min(mX_v , mX_v3), mX_v4);

        mX_out0 = mX.select<BLKH,1,BLKW,1>(0,0);
        mX_out1 = mX.select<BLKH,1,BLKW,1>(1,0);
        mX_out2 = mX.select<BLKH,1,BLKW,1>(2,0);
        mX_out3 = mX.select<BLKH,1,BLKW,1>(3,0);
        mX_out4 = mX.select<BLKH,1,BLKW,1>(4,0);

        mX_out_v = min(min(min(min(mX_out_v0 , mX_out_v1), mX_out_v2),mX_out_v3),mX_out_v4);
        // mX_out_v = min(min(mX_out_v , mX_out_v3), mX_out_v4);

        media_block_store<uchar, BLKH, BLKW>(output_acc, h_pos, v_pos, mX_out);
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

    std::cout << "GPU kernel erosion time=" << kernel_time << "ms\n";

    return;
}