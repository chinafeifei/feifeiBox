//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================

#include "ippcv.h"
#include "ipps.h"
#include "ippcore.h"

#include <iostream>
#include <stdio.h>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc.hpp> // gaussian blur
#include <opencv2/core/utils/logger.hpp>

#include <tbb/tbb.h>
using namespace tbb;

using millis = std::chrono::milliseconds;
using micros = std::chrono::microseconds;

using namespace std;
using namespace cv;

static const bool DEBUG = false;

static const bool INFO = false;

typedef IppStatus(*ippiFilterGaussian_8u)(const Ipp8u* pSrc, IppSizeL srcStep,
    Ipp8u* pDst, IppSizeL dstStep, IppiSizeL roiSize, IppiBorderType borderType, const Ipp8u borderValue[3],
    IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer);

void displayArray(Ipp8u array[], int row, int col, int channel)
{
    if (!DEBUG)
        return;
    cout << "row is " << row << endl;
    int step = col * channel;
    for (int x = 0; x < row; x++)
    {
        for (int y = 0; y < col * channel; y += 3)
        {
            printf("[%3u, %3u, %3u], ", array[x * step + y], array[x * step + y + 1], array[x * step + y + 2]);
        }
        printf("\n");
    }
    printf("\n");
}

void compare(Ipp8u pSrc[], cv::Mat &smoothed)
{
    Ipp8u *pDst = smoothed.data;
    int step = smoothed.cols * smoothed.channels();
    int same = 0;
    int diff = 0;
    for (int x = 0; x < smoothed.rows; x++)
    {
        for (int y = 0; y < smoothed.cols * smoothed.channels(); y++)
        {
            if (pSrc[x * step + y] == pDst[x * step + y])
                same++;
            else
            {
                diff++;
            }
        }
    }

    float fsame = same;
    float fall = same + diff;

    float correctness = fsame / fall * 100;

    // printf("pixels same is %d, diff is %d \n", same, diff);
    if (INFO) {
        printf("pixels same is %d, diff is %d \n", same, diff);
        printf("pixels correctness is %2.2f %% \n", correctness);
    }
}

long long gaussian_opencv(int ksize, double sigma, cv::Mat &img, cv::Mat &smoothed)
{
    if (INFO)
        printf("\n\n=======  opencv gaussian blur START ===========\n");
    auto start = std::chrono::high_resolution_clock::now();

    // int ksize = 5;
    // int sigma = 3;

    GaussianBlur(img, smoothed, Size(ksize, ksize), sigma, sigma, BORDER_REPLICATE);

    auto stop = std::chrono::high_resolution_clock::now();
    auto opencv_time = std::chrono::duration_cast<micros>(stop - start).count();

    if (INFO)
        printf("opencv gaussian filter took %ld microseconds \n", opencv_time);

    if (DEBUG)
    {
        cout << "smoothed (grad_x) = \n"
             << format(smoothed, Formatter::FMT_C) << ";" << endl
             << endl
             << endl;
    }

    return opencv_time;
}

long long gaussian_tbb_ipp_platform_aware(Ipp32u kernelSize, Ipp32f sigma, cv::Mat &img, cv::Mat &smoothed)
{
    if (INFO)
        printf("\n\n=======  tbb ipp platform aware gaussian START ===========\n");
    Ipp8u *pSrc = img.data;
    Ipp8u *pDst = new Ipp8u[img.rows * img.cols * img.channels()]();
    Ipp8u *pBuffer;
    IppFilterGaussianSpec *pSpec;
    Ipp8u *pInitBuffer;

    const int THREAD_NUM = 4;                // tbb 线程 slot 数量，在 4 核 8 线程的 cpu 上，该值最多为8
                                             // 如果设置为 8 ，而 parallel_for 中的 blocked_range 为 0 ，WORKLOAD_NUM = 4
                                             // 那么将会有 4 个 tbb 线程在工作，另外 4 个 tbb 线程空闲
    const int WORKLOAD_NUM = 4;              // 将计算负载分割为多少份
    int chunksize = img.rows / WORKLOAD_NUM; // 每个线程计算区域的行数

    if (INFO)
        printf("chunksize is %d \n", chunksize);

    IppiSizeL roiSize = {img.cols, chunksize}; // 每一个线程计算的区域大小
    IppiBorderType borderType = ippBorderRepl;
    IppSizeL srcStep = img.cols * sizeof(Ipp8u) * img.channels(); // 每一行有多少个字节
    IppSizeL dstStep = img.cols * sizeof(Ipp8u) * img.channels();
    // Ipp8u borderValues[] = { 0, 0, 0 };

    Ipp8u borderValue[] = {0, 0, 0};
    Ipp64f borderVal[4];
    // Ipp32f sigma = 3.0;

    IppSizeL specSize = 0, initSize = 0, bufferSize = 0;
    IppStatus status;
    // Ipp32u kernelSize = 5;

    status = ippiFilterGaussianGetSpecSize_L(kernelSize, ipp8u, img.channels(), &specSize, &initSize);
    if (status < 0)
    {
        printf("Error, ippiFilterGaussianGetSpecSize_L");
        return -1;
    }

    pSpec = (IppFilterGaussianSpec *)ippsMalloc_8u_L(specSize);
    pInitBuffer = ippsMalloc_8u_L(initSize);

    status = ippiFilterGaussianInit_L(roiSize, kernelSize, sigma, borderType, ipp8u, img.channels(), pSpec, pInitBuffer);
    if (status < 0)
    {
        printf("Error, ippiFilterGaussianInit_L %d \n", status);
        return -1;
    }

    status = ippiFilterGaussianGetBufferSize_L(roiSize, kernelSize, ipp8u, borderType, img.channels(), &bufferSize);
    if (status < 0)
    {
        printf("Error, ippiFilterGaussianGetBufferSize_L %d \n", status);
        return -1;
    }

    Ipp8u *pBufferArray[WORKLOAD_NUM];

    for (int i = 0; i < WORKLOAD_NUM; i++)
    {
        pBufferArray[i] = ippsMalloc_8u_L(bufferSize);
    }
    static affinity_partitioner ap;

    // 使用函数指针切换不同通道的计算版本
    ippiFilterGaussian_8u calc;
    if (img.channels() == 3)
        calc = ippiFilterGaussian_8u_C3R_L;
    else
        calc = ippiFilterGaussian_8u_C1R_L;

    // for (int jdx = 0; jdx < 50; jdx++)
    // {
    auto start = std::chrono::high_resolution_clock::now();

    tbb::task_arena no_hyper_thread_arena(
        tbb::task_arena::constraints{}
            .set_max_threads_per_core(1)       // 限制每个物理核只能运行一个 tbb 线程，避免超线程影响
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
                                                         pSrcT = pSrc + srcStep * chunksize * i;
                                                         pDstT = pDst + dstStep * chunksize * i;
                                                         IppiBorderType tBorder;

                                                         if (i == 0)
                                                         {
                                                             tBorder = static_cast<IppiBorderType>(borderType | ippBorderInMemBottom);
                                                         }
                                                         else if (i < WORKLOAD_NUM - 1)
                                                         {
                                                             tBorder = static_cast<IppiBorderType>(borderType | ippBorderInMemTop | ippBorderInMemBottom);
                                                         }
                                                         else
                                                         {
                                                             tBorder = static_cast<IppiBorderType>(borderType | ippBorderInMemTop);
                                                         }
                                                         auto start = std::chrono::high_resolution_clock::now();
                                                         tStatus = calc(pSrcT, srcStep, pDstT, dstStep, roiSize, tBorder, borderValue, pSpec, pBufferArray[i]);
                                                         //  tStatus = ippiFilterGaussian_8u_C3R_L(pSrc, srcStep, pDst, dstStep, roiSize, borderType, borderValue, pSpec, pBufferArray[0]);
                                                         if (tStatus < 0)
                                                         {
                                                             printf("Error, ippiFilterGaussian_8u_C3R_L %d \n", status);
                                                         }
                                                         auto stop = std::chrono::high_resolution_clock::now();
                                                         auto tTime = std::chrono::duration_cast<millis>(stop - start).count();
                                                         if (DEBUG)
                                                         {
                                                             std::cout << "thread " << tbb::this_task_arena::current_thread_index() << " idx " << i << "  took " << tTime << " milliseconds" << endl;
                                                         }
                                                         //  printf("idx is %zu \n", i);
                                                     }
                                                 }); });
    auto stop = std::chrono::high_resolution_clock::now();
    auto tbb_time = std::chrono::duration_cast<micros>(stop - start).count();
    if (INFO)
        printf("tbb ipp platform aware gaussian filter took %ld microseconds \n", tbb_time);

    displayArray(pDst, img.rows, img.cols, img.channels());

    compare(pDst, smoothed);

    delete[] pDst;
    return tbb_time;
}

long long gaussian_ipp_platform_aware(Ipp32u kernelSize, Ipp32f sigma, cv::Mat &img, cv::Mat &smoothed)
{
    if (INFO)
        printf("\n\n=======  ipp platform aware gaussian START ===========\n");
    Ipp8u *pSrc = img.data;
    Ipp8u *pDst = new Ipp8u[img.rows * img.cols * img.channels()]();
    Ipp8u *pBuffer;
    IppFilterGaussianSpec *pSpec;
    Ipp8u *pInitBuf;

    IppiSizeL roiSize = {img.cols, img.rows};
    IppiBorderType borderType = ippBorderRepl;
    IppSizeL srcStep = img.cols * sizeof(Ipp8u) * img.channels(); // 每一行有多少个字节
    IppSizeL dstStep = img.cols * sizeof(Ipp8u) * img.channels();
    // Ipp8u borderValues[] = { 0, 0, 0 };

    Ipp8u borderValues[] = {0, 0, 0};
    Ipp64f borderVal[4];
    // Ipp32f sigma = 3.0;

    // int bufferSize;
    // Ipp32s specSize = 0, initSize=0, bufferSize = 0;
    IppSizeL specSize = 0, initSize = 0, bufferSize = 0;
    IppStatus status;
    // Ipp32u kernelSize = 5;

    status = ippiFilterGaussianGetSpecSize_L(kernelSize, ipp8u, img.channels(), &specSize, &initSize);
    if (status < 0)
    {
        printf("Error, ippiFilterGaussianGetSpecSize_L");
        return -1;
    }

    pSpec = (IppFilterGaussianSpec *)ippMalloc(specSize);
    pInitBuf = ippsMalloc_8u(initSize);

    status = ippiFilterGaussianInit_L(roiSize, kernelSize, sigma, borderType, ipp8u, img.channels(), pSpec, pInitBuf);
    if (status < 0)
    {
        printf("Error, ippiFilterGaussianInit_L, status: %d \n", status);
        return -1;
    }

    status = ippiFilterGaussianGetBufferSize_L(roiSize, kernelSize, ipp8u, borderType, img.channels(), &bufferSize);

    if (INFO)
        printf("buffer size is %lld \n", bufferSize);

    pBuffer = ippsMalloc_8u(bufferSize);

    // 使用函数指针切换不同通道的计算版本
    ippiFilterGaussian_8u calc;
    if (img.channels() == 3)
        calc = ippiFilterGaussian_8u_C3R_L;
    else
        calc = ippiFilterGaussian_8u_C1R_L;


    auto start = std::chrono::high_resolution_clock::now();

    status = calc(pSrc, srcStep, pDst, dstStep, roiSize, borderType, borderValues, pSpec, pBuffer);

    if (status < 0)
    {
        printf("Error, ippiFilterGaussian_8u_C3R_L, status: %d \n", status);
        printf("ippStsNoErr \t %d \n", ippStsNoErr);
        printf("ippStsNullPtrErr \t %d \n", ippStsNullPtrErr);
        printf("ippStsSizeErr \t %d \n", ippStsSizeErr);
        printf("ippStsStepErr \t %d \n", ippStsStepErr);
        printf("ippStsNotEvenStepErr \t %d \n", ippStsNotEvenStepErr);
        printf("ippStsBorderErr \t %d \n", ippStsBorderErr);
        printf("ippStsBadArgErr \t %d \n", ippStsBadArgErr);
        return -1;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto ippL_time = std::chrono::duration_cast<micros>(stop - start).count();

    if (INFO)
        printf("ipp platform aware gaussian filter took %ld microseconds \n", ippL_time);

    displayArray(pDst, img.rows, img.cols, 3);

    compare(pDst, smoothed);

    delete[] pDst;
    return ippL_time;
}

void test(bool isIpp, bool isSingle, Ipp32u kernelSize, Ipp32f sigma, cv::Mat &img, cv::Mat &smoothed, int test_num)
{

    float averageTime = 0.0f;
    long long consumeTime = 0;
    const char *mode = "null";
    for (int i = 0; i < test_num; i++)
    {
        if (isIpp)
        { // ipp method
            if (isSingle)
            {
                mode = "ipp single thread";
                consumeTime = gaussian_ipp_platform_aware(kernelSize, sigma, img, smoothed);
            }
            else
            {
                mode = "ipp multiple thread";
                consumeTime = gaussian_tbb_ipp_platform_aware(kernelSize, sigma, img, smoothed);
            }
        }
        else
        { // opencv method
            if (isSingle)
            {
                mode = "opencv single thread";
                cv::setNumThreads(0);
                consumeTime = gaussian_opencv(kernelSize, sigma, img, smoothed);
            }
            else
            {
                mode = "opencv multiple thread";
                consumeTime = gaussian_opencv(kernelSize, sigma, img, smoothed);
            }
        }
        if (i != 0)
        {
            averageTime += consumeTime/1000.0;
            if(INFO)
                printf("%.2f ms, ", consumeTime/1000.0);
        }
    }

    averageTime /= test_num - 1;
    printf("\n%s kernelSize %d sigma %.1f average time %.2f ms \n\n\n", mode, kernelSize, sigma, averageTime);
}

int main(int argc, char **argv)
{
    const int TEST_NUM = 100;
    bool SINGLE_MODE = false;

    int arg;

    std::string filename = "data/color_4288.jpg";

    cout << "image file is " << filename << endl;
    //Mat img = imread(filename);
    Mat img = imread(filename , IMREAD_GRAYSCALE);

    // Mat img = Mat(10, 10, CV_8UC3, Scalar(0, 0, 0));
    // img.at<cv::Vec3b>(4, 4)[0] = 255;
    // img.at<cv::Vec3b>(4, 4)[1] = 255;

    if (DEBUG)
    {
        cout << "img (grad_x) = \n"
             << format(img, Formatter::FMT_C) << ";" << endl
             << endl

             << endl;
    }

    Mat smoothed;

    // https://docs.opencv.org/4.x/da/db0/namespacecv_1_1utils_1_1logging.html
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);



    // opencv multiple thread kernel 3 sigma 1.0
    test(false, false, 3, 1.0, img, smoothed, TEST_NUM);

    // opencv multiple thread kernel 5 sigma 3.0
    test(false, false, 5, 3.0, img, smoothed, TEST_NUM);

    // opencv multiple thread kernel 21 sigma 4.3
    test(false, false, 21, 4.3, img, smoothed, TEST_NUM);


    printf("================================================================================\n");


    // opencv single thread kernel 3 sigma 1.0
    test(false, true, 3, 1.0, img, smoothed, TEST_NUM);

    // opencv single thread kernel 5 sigma 3.0
    test(false, true, 5, 3.0, img, smoothed, TEST_NUM);

    // opencv single thread kernel 21 sigma 4.3
    test(false, true, 21, 4.3, img, smoothed, TEST_NUM);

    printf("================================================================================\n");

    // ipp single thread kernel 3 sigma 1.0
    test(true, true, 3, 1.0, img, smoothed, TEST_NUM);

    
    // ipp single thread kernel 5 sigma 3.0
    test(true, true, 5, 3.0, img, smoothed, TEST_NUM);

    // ipp single thread kernel 21 sigma 4.3
    test(true, true, 21, 4.3, img, smoothed, TEST_NUM);

    printf("================================================================================\n");

    // ipp multiple thread kernel 3 sigma 1.0
    test(true, false, 3, 1.0, img, smoothed, TEST_NUM);

    // ipp multiple thread kernel 5 sigma 3.0
    test(true, false, 5, 3.0, img, smoothed, TEST_NUM);

    // ipp multiple thread kernel 21 sigma 4.3
    test(true, false, 21, 4.3, img, smoothed, TEST_NUM);

    printf("================================================================================\n");





    
}
