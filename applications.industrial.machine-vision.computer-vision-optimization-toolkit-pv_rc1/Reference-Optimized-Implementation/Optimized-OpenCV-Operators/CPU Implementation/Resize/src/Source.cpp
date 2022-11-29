//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include <cstdio>
#include <iostream>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <omp.h>

#include "ipp.h"
#include "ippcore_tl.h"
#include "ippi_tl.h"
#include "ipps.h"
#include "ippcore_tl.h"

//#pragma comment(lib, "ippcoremt_tl_omp.lib")
//#pragma comment(lib, "ippimt_tl_omp.lib")

#define MAX_NUM_THREADS 8

using namespace cv;
using namespace std;
using namespace std::chrono;

void resize_openCV(Mat, int, int, int);
void resize_ipp(Mat, int, int, int);
void resize_tl(Mat, int, int, int);

int main(int argc, char *argv[])
{
    const int loop = 100;

    Mat img = imread("./data/image2.jpg", cv::IMREAD_UNCHANGED);
    int resize_h = img.rows / 2;
    int resize_w = img.cols / 2;

    if (img.empty())
    {
        cout << "Load failed" << endl;
        return -1;
    }

    imshow("raw", img);
    waitKey(1000);
    destroyAllWindows();

    // openCV
    resize_openCV(img, resize_w, resize_h, loop);

    // ipp
    resize_ipp(img, resize_w, resize_h, loop);

    // tl
    resize_tl(img, resize_w, resize_h, loop);

    // system("pause");

    return 0;
}

void resize_openCV(Mat img, int resize_w, int resize_h, int loop)
{
    Size dsize = Size(resize_w, resize_h);
    Mat dst_tmp = Mat(resize_h, resize_w, CV_8UC3);

    float sum = 0;
    high_resolution_clock::time_point start, end;
    duration<double, std::milli> delta;

    for (int i = 0; i < loop; i++)
    {
        start = high_resolution_clock::now();
        resize(img, dst_tmp, dsize, 0, 0, INTER_NEAREST);
        end = high_resolution_clock::now();
        delta = end - start;
        sum += delta.count();
    }

    imshow("openCV", dst_tmp);
    waitKey(1000);
    destroyAllWindows();

    cout << "The openCV resize time is: " << sum << "ms" << endl;
    cout << "The Average resize time: " << sum / loop << "ms" << endl
         << endl;
}

void resize_ipp(Mat img, int resize_w, int resize_h, int loop)
{
    IppiSize srcSize, dstSize;
    srcSize.width = img.cols;
    srcSize.height = img.rows;
    dstSize.width = resize_w;
    dstSize.height = resize_h;

    Mat dst_tmp = Mat(resize_h, resize_w, CV_8UC3);
    int srcStep = img.step;
    int dstStep = dst_tmp.step;
    Ipp8u *pSrc = (Ipp8u *)&img.data[0];
    Ipp8u *pDst = (Ipp8u *)&dst_tmp.data[0];

    int specSize = 0;
    int initSize = 0;
    Ipp32u numChannels = 3;
    IppStatus ippStatus;

    int g_bufSize = 0;
    IppiResizeSpec_32f *g_pSpec = 0;

    ippStatus = ippiResizeGetSize_8u(srcSize, dstSize, ippNearest, 0, &specSize, &initSize);
    g_pSpec = (IppiResizeSpec_32f *)ippsMalloc_8u(specSize);
    ippStatus = ippiResizeNearestInit_8u(srcSize, dstSize, g_pSpec);
    ippStatus = ippiResizeGetBufferSize_8u(g_pSpec, dstSize, numChannels, &g_bufSize);

    Ipp8u *pBuffer = 0;
    IppiPoint dstOffset = {0, 0};
    pBuffer = ippsMalloc_8u(g_bufSize);

    float sum = 0;
    high_resolution_clock::time_point start, end;
    duration<double, std::milli> delta;

    for (int i = 0; i < loop; i++)
    {
        start = high_resolution_clock::now();
        ippStatus = ippiResizeNearest_8u_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, g_pSpec, pBuffer);
        end = high_resolution_clock::now();
        delta = end - start;
        sum += delta.count();
    }

    imshow("ipp", dst_tmp);
    waitKey(1000);
    destroyAllWindows();

    cout << "The IPP resize time is: " << sum << "ms" << endl;
    cout << "The Average resize time: " << sum / loop << "ms" << endl
         << endl;

    if (pBuffer)
    {
        ippsFree(pBuffer);
    }
    if (g_pSpec)
    {
        ippsFree(g_pSpec);
    }
}

void resize_tl(Mat img, int resize_w, int resize_h, int loop)
{
    IppiSizeL srcSize_TL, dstSize_TL;
    srcSize_TL.width = img.cols;
    srcSize_TL.height = img.rows;
    dstSize_TL.width = resize_w;
    dstSize_TL.height = resize_h;

    Mat dst_tmp = Mat(resize_h, resize_w, CV_8UC3);
    Ipp32s srcStep = img.step;
    Ipp32s dstStep = dst_tmp.step;
    Ipp8u *pSrc = (Ipp8u *)&img.data[0];
    Ipp8u *pDst = (Ipp8u *)&dst_tmp.data[0];

    IppSizeL specSize = 0;
    IppSizeL tempSize = 0;
    Ipp32u numChannels = 3;
    IppStatus ippStatus;

    IppiResizeSpec_LT *g_pSpec_TL = 0;

    int threads = 8;
    omp_set_num_threads(threads);

    ippStatus = ippiResizeGetSize_LT(srcSize_TL, dstSize_TL, ipp8u, ippNearest, 0, &specSize, &tempSize);
    g_pSpec_TL = (IppiResizeSpec_LT *)ippsMalloc_8u(specSize);
    ippStatus = ippiResizeNearestInit_LT(srcSize_TL, dstSize_TL, ipp8u, numChannels, g_pSpec_TL);
    ippStatus = ippiResizeGetBufferSize_LT(g_pSpec_TL, &tempSize);
    Ipp8u *pBuffer = ippsMalloc_8u(tempSize);

    float sum = 0;
    high_resolution_clock::time_point start, end;
    duration<double, std::milli> delta;

    for (int i = 0; i < loop; i++)
    {
        start = high_resolution_clock::now();
        ippStatus = ippiResizeNearest_8u_C3R_LT(pSrc, srcStep, pDst, dstStep, g_pSpec_TL, pBuffer);
        end = high_resolution_clock::now();
        delta = end - start;
        sum += delta.count();
    }

    imshow("TL", dst_tmp);
    waitKey(1000);
    destroyAllWindows();

    cout << "The TL resize time: " << sum << "ms" << endl;
    cout << "The Average resize time: " << sum / loop << "ms" << endl
         << endl;
}