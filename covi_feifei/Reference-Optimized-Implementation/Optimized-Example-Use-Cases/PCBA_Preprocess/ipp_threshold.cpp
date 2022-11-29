//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc.hpp>

#include "ippi.h"

#include <tbb/tbb.h>
#include <omp.h>

#include "util.h"

using namespace tbb;

#include <iostream>

using namespace std;

const bool DEBUG = false;

void ipp_threshold(cv::Mat &image, cv::Mat &output, Ipp8u threshold)
{

    output.create(image.size(), image.type());

    // printf("image step is %d \n", image.step[0]);  // 矩阵第一行元素的字节数

    IppiSize roi = {image.cols, image.rows};

    ippiThreshold_LTValGTVal_8u_C1R(image.data, image.step[0], output.data, output.step[0], roi, threshold, 0, threshold, 255);
}

void ipp_threshold_tbb(cv::Mat &image, cv::Mat &output, Ipp8u threshold)
{
    const int THREAD_NUM = 4;                  // tbb 线程 slot 数量，在 4 核 8 线程的 cpu 上，该值最多为8
                                               // 如果设置为 8 ，而 parallel_for 中的 blocked_range 为 0 ，WORKLOAD_NUM = 4
                                               // 那么将会有 4 个 tbb 线程在工作，另外 4 个 tbb 线程空闲
    const int WORKLOAD_NUM = 4;                // 将计算负载分割为多少份
    int chunksize = image.rows / WORKLOAD_NUM; // 每个线程计算区域的行数  2654/4 = 663 余 2
    int remainder = image.rows % WORKLOAD_NUM;

    output.create(image.size(), image.type());

    if (DEBUG)
        printf("chunksize is %d, remainder is %d \n", chunksize, remainder);

    IppiSize roiSize = {image.cols, chunksize};                 // 每一个线程计算的区域大小
    IppiSize lastRoiSize = {image.cols, chunksize + remainder}; // 由于图片行数无法除尽 4，最后一个区域的行数需要把余数加上

    tbb::task_arena no_hyper_thread_arena(
        tbb::task_arena::constraints{}
            .set_max_threads_per_core(1)       // 限制每个物理核只能运行一个 tbb 线程，避免超线程影响，这个功能需要添加预处理定义 __TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION_PRESENT
            .set_max_concurrency(THREAD_NUM)); // 设置 tbb 线程 slot 数量

    no_hyper_thread_arena.execute([&]
                                  { parallel_for(blocked_range<size_t>(0, WORKLOAD_NUM, 1),
                                                 [&](const blocked_range<size_t> &r)
                                                 {
                                                     // 这里一共有 WORKLOAD_NUM 个循环，由 tbb 自动分配给相应的线程执行
                                                     for (size_t i = r.begin(); i != r.end(); ++i)
                                                     {
                                                         //  IppStatus tStatus;

                                                         Ipp8u *pSrcT; // 分别指向原矩阵的不同起始地址
                                                         Ipp8u *pDstT; // 分别指向目标矩阵的不同起始地址
                                                         IppStatus tStatus;
                                                         pSrcT = image.data + image.step[0] * chunksize * i;
                                                         pDstT = output.data + output.step[0] * chunksize * i;
                                                         IppiBorderType tBorder;
                                                         IppiSize sliceRoiSize;

                                                         if (i == WORKLOAD_NUM - 1)
                                                         {
                                                             sliceRoiSize = lastRoiSize;
                                                         }
                                                         else
                                                         {
                                                             sliceRoiSize = roiSize;
                                                         }

                                                         // tStatus = calc(pSrcT, srcStep, pDstT, dstStep, roiSize, tBorder, borderValue, pSpec, pBufferArray[i]);

                                                         tStatus = ippiThreshold_LTValGTVal_8u_C1R(pSrcT, image.step[0], pDstT, output.step[0], sliceRoiSize, threshold, 0, threshold, 255);

                                                         //  tStatus = ippiFilterGaussian_8u_C3R_L(pSrc, srcStep, pDst, dstStep, roiSize, borderType, borderValue, pSpec, pBufferArray[0]);
                                                         if (tStatus < 0)
                                                         {
                                                             printf("Error, ippiFilterGaussian_8u_C3R_L %d \n", tStatus);
                                                         }
                                                         // auto stop = std::chrono::high_resolution_clock::now();
                                                         // auto tTime = std::chrono::duration_cast<millis>(stop - start).count();
                                                         // if (DEBUG)
                                                         //{
                                                         //     std::cout << "thread " << tbb::this_task_arena::current_thread_index() << " idx " << i << "  took " << tTime << " milliseconds" << endl;
                                                         // }
                                                         //   printf("idx is %zu \n", i);
                                                     }
                                                 }); });
}

void ipp_threshold_omp(cv::Mat &image, cv::Mat &output, Ipp8u threshold)
{

    const int WORKLOAD_NUM = 4;                // 将计算负载分割为多少份
    int chunksize = image.rows / WORKLOAD_NUM; // 每个线程计算区域的行数  2654/4 = 663 余 2
    int remainder = image.rows % WORKLOAD_NUM;
    output.create(image.size(), image.type());

    IppiSize roiSize = {image.cols, chunksize};                 // 每一个线程计算的区域大小
    IppiSize lastRoiSize = {image.cols, chunksize + remainder}; // 由于图片行数无法除尽 4，最后一个区域的行数需要把余数加上

    //#pragma omp parallel num_threads(WORKLOAD_NUM)
    //    {
    //#pragma omp master
    //        {
    //            int numThreads = omp_get_num_threads();
    //        }
    //
    //#pragma omp barrier
    //        {
    //
    //            {
    //                /*Split images to subimages and operate on subimages*/
    //                Ipp8u* pSrcT;
    //                Ipp8u* pDstT;
    //                IppStatus tStatus;
    //                IppiSize sliceRoiSize = roiSize;
    //                int i = omp_get_thread_num();
    //
    //                kmp_affinity_mask_t mask;
    //                kmp_create_affinity_mask(&mask);
    //                kmp_set_affinity_mask_proc(2 * i, &mask);
    //                kmp_set_affinity(&mask);
    //
    //                if (i == WORKLOAD_NUM - 1)
    //                    sliceRoiSize = lastRoiSize;
    //
    //                pSrcT = image.data + image.step[0] * chunksize * i;
    //                pDstT = output.data + output.step[0] * chunksize * i;
    //                tStatus = ippiThreshold_LTValGTVal_8u_C1R(pSrcT, image.step[0], pDstT, output.step[0], sliceRoiSize, 200, 0, 200, 255);
    //            }
    //        }
    //    }

#pragma omp parallel num_threads(WORKLOAD_NUM)
    {
        // cout << "num_threads()" << GetCurrentThread() << endl;
        int ompTid = omp_get_thread_num();
        kmp_affinity_mask_t mask;
        kmp_create_affinity_mask(&mask);
        kmp_set_affinity_mask_proc(2 * ompTid, &mask);
        kmp_set_affinity(&mask);
    }
#pragma omp parallel for num_threads(WORKLOAD_NUM)
    // creation of paralel threads
    for (int i = 0; i < WORKLOAD_NUM; ++i)
    { /* For every odd number */ // comparing with Pthread file i didnt

        Ipp8u *pSrcT;
        Ipp8u *pDstT;
        IppStatus tStatus;
        IppiSize sliceRoiSize = roiSize;
        if (i == WORKLOAD_NUM - 1)
            sliceRoiSize = lastRoiSize;

        pSrcT = image.data + image.step[0] * chunksize * i;
        pDstT = output.data + output.step[0] * chunksize * i;
        tStatus = ippiThreshold_LTValGTVal_8u_C1R(pSrcT, image.step[0], pDstT, output.step[0], sliceRoiSize, threshold, 0, threshold, 255);
    }
}

void ipp_threshold_omp_otsu(cv::Mat &image, cv::Mat &output)
{

    Ipp8u otsuThreshold = 0;
    ippiComputeThreshold_Otsu_8u_C1R(image.data, image.step[0], IppiSize{image.cols, image.rows}, &otsuThreshold);
    ipp_threshold_omp(image, output, otsuThreshold);
}

/**
 * 参考 https://zj-image-processing.readthedocs.io/zh_CN/latest/opencv/code/[threshold]%E5%9F%BA%E6%9C%AC%E9%98%88%E5%80%BC%E6%93%8D%E4%BD%9C/
 *       https://docs.opencv.org/4.6.0/db/d8e/tutorial_threshold.html
 *
 * THRESH_BINARY     如果像素比阈值大，就设置为最大值，否则设为 0
 * THRESH_TRUNC      如果像素比阈值大，就设置为最大值，否则保留像素值
 * THRESH_TOZERO     如果像素比阈值大，就保留像素值，否则设为 0
 * THRESH_TOZERO_INV 如果像素值比阈值大，就设为 0，否则保留像素值
 * THRESH_OTSU       使用 OTSU 算法计算最佳阈值
 *
 *
 */

int main_ipp_threshold(int argc, char **argv)
// int main(int argc, char** argv)
{
    // std::string filename = "data/so_small_starry_25.png";
    // std::string filename = "data/color_4288.jpg";
    // std::string filename = "data/example_gray.bmp";

    std::string filename = "data/1k.jpg";

    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    cv::Mat output;

    cv::Mat ippOutput;

    displayArray(image, DEBUG);

    const int LOOP_NUM = 50;

    float average = 0.0f;

    for (int i = 0; i < LOOP_NUM; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::threshold(image, output, 200, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        auto stop = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
        average += time;
        printf("opencv time is %.3f milliseconds\n", time);
    }

    printf("OPENCV average time is %.3f milliseconds \n", average / LOOP_NUM);
    printf("output is \n");

    displayArray(output, DEBUG);

    average = 0.0f;
    for (int i = 0; i < LOOP_NUM; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        Ipp8u otsuThreshold = 0;
        ippiComputeThreshold_Otsu_8u_C1R(image.data, image.step[0], IppiSize{image.cols, image.rows}, &otsuThreshold);
        ipp_threshold(image, ippOutput, otsuThreshold);
        auto stop = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;

        average += time;
        printf("ipp single time is %.3f milliseconds\n", time);
    }

    printf("IPP average time is %.3f milliseconds \n", average / LOOP_NUM);
    printf("ipp output is \n");
    displayArray(ippOutput, DEBUG);

    average = 0.0f;
    for (int i = 0; i < LOOP_NUM; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        ipp_threshold_omp_otsu(image, ippOutput);
        auto stop = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;

        average += time;
        printf("ipp omp time is %.3f milliseconds\n", time);
    }
    printf("IPP omp average time is %.3f milliseconds \n", average / LOOP_NUM);

    average = 0.0f;
    for (int i = 0; i < LOOP_NUM; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        ipp_threshold_omp_otsu(image, ippOutput);
        auto stop = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;

        average += time;
        printf("ipp omp time is %.3f milliseconds\n", time);
    }
    printf("IPP omp average time is %.3f milliseconds \n", average / LOOP_NUM);

    comparecv(output, ippOutput);
}