//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================

#include "ipp.h"
#include "ippi.h"
#include "ippcc.h"
#include "ipptypes.h"
#include "ippi_tl.h"
#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

static const bool DEBUG = false;

void log(const char *txt)
{
    if (DEBUG)
        printf("%s \n", txt);
}

void compare(Ipp16s *single_dst, Ipp16s *multi_dst, cv::Mat &cv_dst)
{
    using milli = std::chrono::milliseconds;

    auto compare_start = std::chrono::high_resolution_clock::now();

    bool isTrue = true;
    bool isMultiFail = false;
    bool isSingleFail = false;

    for (int x = 0; x < cv_dst.rows; x++)
    {
        for (int y = 0; y < cv_dst.cols; y++)
        {
            if (multi_dst[x * cv_dst.cols + y] != cv_dst.at<short>(x, y))
            {
                isTrue = false;
                isMultiFail = true;
                break;
            }
            if (multi_dst[x * cv_dst.cols + y] != single_dst[x * cv_dst.cols + y])
            {
                isTrue = false;
                isSingleFail = true;
                break;
            }
        }
    }

    if (isTrue)
    {
        cout << "================  compare pass  ======================" << endl;
    }
    else
    {
        cout << "================  compare fail  ======================" << endl;
        if (!isMultiFail)
        {
            cout << "  Multi Sobel fail compared to opencv result" << endl;
        }
        if (!isSingleFail)
        {
            cout << "  Single Sobel fail compared to opencv result" << endl;
        }
    }

    auto compare_finish = std::chrono::high_resolution_clock::now();

    std::cout << "compare took "
              << std::chrono::duration_cast<milli>(compare_finish - compare_start).count()
              << " milliseconds\n";
}

template <typename T>
void displayArrayFloat(T array[], int row, int col)
{
    if (!DEBUG)
    {
        return;
    }
    cout << "row is " << row << "  col is " << col << endl;

    for (int x = 0; x < row; x++)
    {
        for (int y = 0; y < col; y++)
        {
            // printf("%7.1f ", array[x * col + y]);
            cout << array[x * col + y];
        }
        cout << endl;
    }
    cout << endl;
}

template <typename T>
void displayArray(T array[], int row, int col)
{
    if (!DEBUG)
    {
        return;
    }
    cout << "row is " << row << "  col is " << col << endl;

    for (int x = 0; x < row; x++)
    {
        for (int y = 0; y < col; y++)
        {
            // printf("%7d ", array[x * col + y]);
            cout << array[x * col + y];
        }
        cout << endl;
    }
    cout << endl;
}

void displayArray(Ipp16s array[], int row, int col, int channel)
{
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

void sobel_ipp_single_thread(cv::Mat &img, Ipp16s *dst)
{
    std::cout << endl
              << endl
              << "=============  SINGLE thread sobel hypot ==================" << endl;

    using milli = std::chrono::milliseconds;

    IppStatus status;
    Ipp8u *pSrc = img.data;

    Ipp32f *fSrc = new Ipp32f[img.rows * img.cols * img.channels()];  // 32f source data
    Ipp32f *hDst = new Ipp32f[img.rows * img.cols * img.channels()];  // horizon  result
    Ipp32f *vDst = new Ipp32f[img.rows * img.cols * img.channels()];  // vertical result
    Ipp32f *hsDst = new Ipp32f[img.rows * img.cols * img.channels()]; // horizon  square result
    Ipp32f *vsDst = new Ipp32f[img.rows * img.cols * img.channels()]; // vertical square result

    int step8 = img.cols * sizeof(Ipp8u);
    int step16 = img.cols * sizeof(Ipp16s);
    int step32 = img.cols * sizeof(Ipp32s);
    IppiBorderType borderType = ippBorderRepl; // border type

    Ipp8u *pBuffer;
    int bufferSizeH = 0; // horizon  convolution buffer size
    int bufferSizeV = 0; // vertical convolution buffer size

    IppiSize roiSize = {img.cols, img.rows};

    // in order to prevent data overflow, change 8u to 32f
    status = ippiConvert_8u32f_C1R(pSrc, step8, fSrc, step32, roiSize);

    auto ipp_start = std::chrono::high_resolution_clock::now();
    ippiFilterSobelVertBorderGetBufferSize(roiSize, ippMskSize3x3, ipp32f, ipp32f, 1, /* numChannels */ &bufferSizeV);
    ippiFilterSobelHorizBorderGetBufferSize(roiSize, ippMskSize3x3, ipp32f, ipp32f, 1, /* numChannels */ &bufferSizeH);

    pBuffer = ippsMalloc_8u(bufferSizeH > bufferSizeV ? bufferSizeH : bufferSizeV); // 创建 horizon 和 vertical 的缓存，取二者的最大值

    log("ipp single horizon sobel");
    status = ippiFilterSobelHorizBorder_32f_C1R(fSrc, step32, hDst, step32,
                                                roiSize, ippMskSize3x3, borderType, 0, pBuffer);
    displayArrayFloat(hDst, img.rows, img.cols);

    log("ipp single vertical sobel");
    status = ippiFilterSobelVertBorder_32f_C1R(fSrc, step32, vDst, step32,
                                               roiSize, ippMskSize3x3, borderType, 0, pBuffer);
    displayArrayFloat(vDst, img.rows, img.cols);

    log("ipp single horizon sobel squre");
    status = ippiSqr_32f_C1R(hDst, step32, hsDst, step32, roiSize);
    displayArrayFloat(hsDst, img.rows, img.cols);

    log("ipp single vertical sobel squre");
    status = ippiSqr_32f_C1R(vDst, step32, vsDst, step32, roiSize);
    displayArrayFloat(vsDst, img.rows, img.cols);

    log("ipp single 2 square add");
    status = ippiAdd_32f_C1IR(hsDst, step32, vsDst, step32, roiSize);
    displayArrayFloat(vsDst, img.rows, img.cols);

    log("ipp single square root");
    status = ippiSqrt_32f_C1IR(vsDst, step32, roiSize);
    displayArrayFloat(vsDst, img.rows, img.cols);

    log("ipp convert 32f to 16s");
    status = ippiConvert_32f16s_C1R(vsDst, step32, dst, step16, roiSize, ippRndFinancial);
    displayArray(dst, img.rows, img.cols);

    auto ipp_finish = std::chrono::high_resolution_clock::now();

    std::cout << "ipp Sobel single thread x took "
              << std::chrono::duration_cast<milli>(ipp_finish - ipp_start).count()
              << " milliseconds\n";

    delete[] hDst;  // horizon  result
    delete[] vDst;  // vertical result
    delete[] hsDst; // horizon  square result
    delete[] vsDst; // vertical square result
    ippsFree(pBuffer);
}

void sobel_ipp_multi_thread(cv::Mat &img, Ipp16s *dst)
{
    using milli = std::chrono::milliseconds;
    std::cout << endl
              << endl
              << "=============  MULTI thread sobel hypot ==================" << endl;

    Ipp8u *pSrc = img.data; // source data

    IppiSize roiSize = {img.cols, img.rows};

    IppStatus statusT;

    int bufferSizeH = 0; // horizon convolution buffer size
    int bufferSizeV = 0; // vertical convolution buffer 大小
    Ipp8u *pBuffer;

    Ipp16s *hDst = new Ipp16s[img.rows * img.cols * img.channels()];  // horizon  result
    Ipp16s *vDst = new Ipp16s[img.rows * img.cols * img.channels()];  // vertical result
    Ipp32s *hsDst = new Ipp32s[img.rows * img.cols * img.channels()]; // horizon  square result
    Ipp32s *vsDst = new Ipp32s[img.rows * img.cols * img.channels()]; // vertical square result

    Ipp16s *hAbsDst = new Ipp16s[img.rows * img.cols * img.channels()];
    Ipp16s *vAbsDst = new Ipp16s[img.rows * img.cols * img.channels()];

    int step8 = img.cols * sizeof(Ipp8u);
    int step16 = img.cols * sizeof(Ipp16s);
    int step32 = img.cols * sizeof(Ipp32s);
    IppiBorderType borderType_T = ippBorderRepl;

    ippiFilterSobelHorizBorderGetBufferSize_T(roiSize, ippMskSize3x3, ipp8u, ipp16s, 1, /* numChannels */ &bufferSizeH);

    auto t_start = std::chrono::high_resolution_clock::now();
    ippiFilterSobelVertBorderGetBufferSize_T(roiSize, ippMskSize3x3, ipp8u, ipp16s, 1, /* numChannels */ &bufferSizeV);

    pBuffer = ippsMalloc_8u(bufferSizeH > bufferSizeV ? bufferSizeH : bufferSizeV);

    // horizon sobel
    statusT = ippiFilterSobelHorizBorder_8u16s_C1R_T(pSrc, step8, hDst, step16,
                                                     roiSize, ippMskSize3x3, borderType_T, 0, pBuffer);

    displayArray(hDst, img.rows, img.cols);

    // vertical sobel
    statusT = ippiFilterSobelVertBorder_8u16s_C1R_T(pSrc, step8, vDst, step16,
                                                    roiSize, ippMskSize3x3, borderType_T, 0, pBuffer);

    log("vertical sobel");
    displayArray(vDst, img.rows, img.cols);

    // horizon square
    statusT = ippiSqr_16s32s_C1RSfs_T(hDst, step16, hsDst, step32, roiSize, 0 /* scale */);

    log("horizon squre");
    displayArray(hsDst, img.rows, img.cols);

    // vertical square
    statusT = ippiSqr_16s32s_C1RSfs_T(vDst, step16, vsDst, step32, roiSize, 0 /* scale */);
    log("vertical squre");
    displayArray(vsDst, img.rows, img.cols);

    // addition
    statusT = ippiAdd_32s_C1IRSfs_T(hsDst, step32, vsDst, step32, roiSize, 0 /* scale */);
    log("horizon squre + vertical squre");
    displayArray(vsDst, img.rows, img.cols);

    // square root
    statusT = ippiSqrt_32s16s_C1RSfs_T(vsDst, step32, dst, step16, roiSize, 0 /* scale */);

    log("squre root");
    displayArray(dst, img.rows, img.cols);

    auto t_finish = std::chrono::high_resolution_clock::now();

    std::cout << "ipp Sobel multi thread x took "
              << std::chrono::duration_cast<milli>(t_finish - t_start).count()
              << " milliseconds\n\n\n";

    delete[] hDst;  // horizon  result
    delete[] vDst;  // vertical result
    delete[] hsDst; // horizon  square result
    delete[] vsDst; // vertical square result
    delete[] hAbsDst;
    delete[] vAbsDst;
    ippsFree(pBuffer);
}

void sobel_opencv(cv::Mat &img, cv::Mat &dst)
{
    using milli = std::chrono::milliseconds;
    std::cout << endl
              << endl
              << "=============  OPENCV sobel hypot =====================" << endl;

    Mat grad_x;
    Mat grad_y;
    Mat square_x;
    Mat square_y;
    Mat add_dst;
    Mat abs_grad_x;
    Mat abs_grad_y;

    int ksize = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;

    auto cv_start = std::chrono::high_resolution_clock::now();
    Sobel(img, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_REPLICATE);
    Sobel(img, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_REPLICATE);

    cv::pow(grad_x, 2, square_x);
    cv::pow(grad_y, 2, square_y);

    cv::add(square_x, square_y, add_dst);
    cv::sqrt(add_dst, dst);
    dst.convertTo(dst, CV_16S);

    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    // addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);

    auto cv_finish = std::chrono::high_resolution_clock::now();

    std::cout << "opencv Sobel x took "
              << std::chrono::duration_cast<milli>(cv_finish - cv_start).count()
              << " milliseconds\n";

    // displayArray(grad_x.data, img.rows, img.cols);

    if (DEBUG)
    {
        cout << "grad_x = \n"
             << format(grad_x, Formatter::FMT_PYTHON) << ";" << endl
             << endl
             << endl;

        cout << "grad_y = \n"
             << format(grad_y, Formatter::FMT_PYTHON) << ";" << endl
             << endl
             << endl;
        cout << "square_x = \n"
             << format(square_x, Formatter::FMT_PYTHON) << ";" << endl
             << endl
             << endl;
        cout << "square_y = \n"
             << format(square_y, Formatter::FMT_PYTHON) << ";" << endl
             << endl
             << endl;
        cout << "add_dst = \n"
             << format(add_dst, Formatter::FMT_PYTHON) << ";" << endl
             << endl
             << endl;
        cout << "dst = \n"
             << format(dst, Formatter::FMT_PYTHON) << ";" << endl
             << endl
             << endl;
    }
}

int main(int argc, char **argv)
{
    using milli = std::chrono::milliseconds;

    std::string filename = "data/color_4288.jpg";

    printf("read file %s \n", filename.c_str());
    Mat img = imread(filename, IMREAD_GRAYSCALE);
    Mat cv_dst;
    Ipp16s *ipp_multi_dst = new Ipp16s[img.rows * img.cols * img.channels()];
    Ipp16s *ipp_single_dst = new Ipp16s[img.rows * img.cols * img.channels()];

    if (DEBUG)
    {
        cout << "img (grad_x) = \n"
             << format(img, Formatter::FMT_C) << ";" << endl
             << endl
             << endl;
    }

    // ======================= opencv sobel =====================
    sobel_opencv(img, cv_dst);

    // ======================= ipp single thread ===========

    sobel_ipp_single_thread(img, ipp_single_dst);

    // ====================== ipp multi thread ==================

    sobel_ipp_multi_thread(img, ipp_multi_dst);

    compare(ipp_single_dst, ipp_multi_dst, cv_dst);

    printf("\n\n");

    delete[] ipp_single_dst;
    delete[] ipp_multi_dst;
    return 0;
}
