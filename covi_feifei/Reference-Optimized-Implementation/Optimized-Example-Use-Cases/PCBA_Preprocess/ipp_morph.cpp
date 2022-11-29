//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc.hpp>

#include "ippi.h"
#include "ipps.h"
#include "ippcv.h"

#include <iostream>

#include <tbb/tbb.h>

#include "util.h"

const bool DEBUG = false;

void ipp_morph_close(cv::Mat &image, cv::Mat &output, cv::Mat &kernel)
{

    output.create(image.size(), image.type());

    IppStatus status;
    Ipp8u *pSrc = image.data;
    Ipp8u *pDst = output.data;
    Ipp8u *kernelPtr = kernel.data;

    IppiSizeL roiSize = {image.cols, image.rows};

    IppiSizeL g_maskSize = {kernel.cols, kernel.rows};

    IppiMorphAdvStateL *m_spec;
    IppSizeL specSize = 0;

    // 获取 spec size 大小
    ippiMorphGetSpecSize_L(roiSize, g_maskSize, ipp8u, image.channels(), &specSize);

    // 创建 spec 内存
    m_spec = (IppiMorphAdvStateL *)malloc((int)(specSize * sizeof(IppiMorphAdvStateL *)));

    // 初始化 spec
    status = ippiMorphInit_L(roiSize, kernelPtr, g_maskSize, ipp8u, image.channels(), m_spec);

    // 创建缓冲区
    Ipp8u *tmpBuffer;
    IppSizeL tmpBufferSize = 0;
    ippiMorphGetBufferSize_L(roiSize, g_maskSize, ipp8u, image.channels(), &tmpBufferSize);
    tmpBuffer = ippsMalloc_8u((int)tmpBufferSize);

    status = ippiMorphClose_8u_C1R_L(pSrc, image.step[0], pDst, output.step[0], roiSize, ippBorderDefault, NULL, m_spec, tmpBuffer);
}

void ipp_morph_close_tbb(cv::Mat &image, cv::Mat &output, cv::Mat &kernel)
{

    output.create(image.size(), image.type());

    const int THREAD_NUM = 4; // tbb 线程 slot 数量，在 4 核 8 线程的 cpu 上，该值最多为8
                              // 如果设置为 8 ，而 parallel_for 中的 blocked_range 为 0 ，WORKLOAD_NUM = 4
                              // 那么将会有 4 个 tbb 线程在工作，另外 4 个 tbb 线程空闲

    const int WORKLOAD_NUM = 4;                // 将计算负载分割为多少份
    int chunksize = image.rows / WORKLOAD_NUM; // 每个线程计算区域的行数  2654/4 = 663 余 2
    int remainder = image.rows % WORKLOAD_NUM;

    IppiSizeL roiSize = {image.cols, chunksize}; // 每一个线程计算的区域大小
    IppiSizeL lastRoiSize = {image.cols, chunksize + remainder};

    IppStatus status;
    Ipp8u *pSrc = image.data;
    Ipp8u *pDst = output.data;
    Ipp8u *kernelPtr = kernel.data;
    IppiSizeL g_maskSize = {kernel.cols, kernel.rows};

    IppiMorphAdvStateL *m_spec;
    IppSizeL specSize = 0;

    // 获取 spec size 大小, 按照最大的 slice 区域来初始化 spec
    ippiMorphGetSpecSize_L(lastRoiSize, g_maskSize, ipp8u, image.channels(), &specSize);

    // 创建 spec 内存
    m_spec = (IppiMorphAdvStateL *)malloc((int)(specSize * sizeof(IppiMorphAdvStateL *)));

    // 初始化 spec，按照最大的 slice 区域来初始化 spec
    status = ippiMorphInit_L(lastRoiSize, kernelPtr, g_maskSize, ipp8u, image.channels(), m_spec);

    // 创建缓冲区，按照最大的 slice 区域来创建缓冲区内存
    IppSizeL tmpBufferSize = 0;
    ippiMorphGetBufferSize_L(lastRoiSize, g_maskSize, ipp8u, image.channels(), &tmpBufferSize);

    Ipp8u *pBufferArray[WORKLOAD_NUM];
    for (int i = 0; i < WORKLOAD_NUM; i++)
    {
        pBufferArray[i] = ippsMalloc_8u((int)tmpBufferSize);
    }

    tbb::task_arena no_hyper_thread_arena(tbb::task_arena::constraints{}.set_max_threads_per_core(1).set_max_concurrency(THREAD_NUM));

    no_hyper_thread_arena.execute([&]
                                  { tbb::parallel_for(tbb::blocked_range<size_t>(0, WORKLOAD_NUM, 1),
                                                      [&](const tbb::blocked_range<size_t> &r)
                                                      {
                                                          // 这里一共有 WORKLOAD_NUM 个循环，由 tbb 自动分配给相应的线程执行
                                                          for (size_t i = r.begin(); i != r.end(); ++i)
                                                          {

                                                              Ipp8u *pSrcT; // 分别指向原矩阵的不同起始地址
                                                              Ipp8u *pDstT; // 分别指向目标矩阵的不同起始地址
                                                              IppStatus tStatus;
                                                              pSrcT = pSrc + image.step[0] * chunksize * i;
                                                              pDstT = pDst + output.step[0] * chunksize * i;
                                                              // tBorder = ippBorderFirstStageInMem | ippBorderDefault | ippBorderInMemTop;
                                                              if (i == 0)
                                                              {
                                                                  tStatus = ippiMorphClose_8u_C1R_L(pSrcT, image.step[0], pDstT, output.step[0], roiSize, IppiBorderType(ippBorderFirstStageInMemBottom | ippBorderDefault), NULL, m_spec, pBufferArray[i]);
                                                              }
                                                              else if (i == WORKLOAD_NUM - 1)
                                                              {
                                                                  tStatus = ippiMorphClose_8u_C1R_L(pSrcT, image.step[0], pDstT, output.step[0], lastRoiSize, IppiBorderType(ippBorderFirstStageInMemTop | ippBorderDefault), NULL, m_spec, pBufferArray[i]);
                                                              }
                                                              else
                                                                  tStatus = ippiMorphClose_8u_C1R_L(pSrcT, image.step[0], pDstT, output.step[0], roiSize, IppiBorderType(ippBorderFirstStageInMem | ippBorderDefault | ippBorderInMemTop | ippBorderInMemBottom), NULL, m_spec, pBufferArray[i]);
                                                          }
                                                      }); });

    delete m_spec;
    for (int i = 0; i < WORKLOAD_NUM; i++)
    {
        ippsFree(pBufferArray[i]);
    }
}

int main_ipp_morph(int argc, char **argv)
// int main(int argc, char** argv)
{
    std::string filename = "data/1k.jpg";
    // std::string filename = "data/example_gray.bmp";
    // std::string filename = "data/so_small_starry_25.png";
    // std::string filename = "data/color_4288.jpg";

    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    // cv::Mat file = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    // cv::Mat image;
    // cv::resize(file, image, cv::Size(), 0.25, 0.25);

    cv::Mat output;

    cv::Mat ippOutput;

    cv::Mat ippTbbOutput;

    displayArray(image, DEBUG);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));

    const int LOOP_NUM = 60;

    float average = 0.0f;

    for (int i = 0; i < LOOP_NUM; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::morphologyEx(image, output, cv::MORPH_CLOSE, kernel);
        auto stop = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
        average += time;
        printf("opencv time is %3.3f milliseconds\n", time);
    }

    printf("OPENCV average time is %.3f milliseconds \n", average / LOOP_NUM);
    printf("output is \n");

    printf("opencv result is \n");
    displayArray(output, DEBUG);

    average = 0.0f;
    for (int i = 0; i < LOOP_NUM; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        ipp_morph_close(image, ippOutput, kernel);

        auto stop = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;

        average += time;
        printf("ipp time is %.3f milliseconds\n", time);
    }

    printf("IPP average time is %.3f milliseconds \n", average / LOOP_NUM);

    average = 0.0f;
    for (int i = 0; i < LOOP_NUM; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        ipp_morph_close_tbb(image, ippTbbOutput, kernel);

        auto stop = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;

        average += time;
        printf("ipp tbb time is %.3f milliseconds\n", time);
    }

    printf("IPP TBB average time is %.3f milliseconds \n", average / LOOP_NUM);

    // comparecv(output, ippOutput);
    comparecv(output, ippTbbOutput);

    printf("\n\n  ipp_morph.cpp finished \n");
}