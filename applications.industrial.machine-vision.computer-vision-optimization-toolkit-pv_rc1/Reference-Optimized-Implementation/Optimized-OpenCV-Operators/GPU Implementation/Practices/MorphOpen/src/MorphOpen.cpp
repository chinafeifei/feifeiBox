//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
// #include<Windows.h>
#include <iostream>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "MorphOpen.h"

using namespace cv;

using time_point = decltype(std::chrono::steady_clock::now());
static inline time_point get_time_point()
{
    // waitForKernelCompletion();
    return std::chrono::steady_clock::now();
}

static inline double get_duration(const time_point &from, const time_point &to)
{
    ;
    return std::chrono::duration<float, std::milli>(to - from).count();
}

void diff(int *himage, int *dimage, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (himage[i * width + j] != dimage[i * width + j])
            {
                std::cout << "Expected: " << himage[i * width + j] << ", actual: " << dimage[i * width + j] << ", on: " << i << ", " << j << std::endl;
                exit(0);
            }
        }
    }
}

int main()
{
    Mat srcMat = imread("./data/color_4288.jpg", 0);
    cv::Mat dstMat = srcMat.clone();
    cv::Mat dstMat_tmp = srcMat.clone();
    cv::Mat dstMat_dpcpp = srcMat.clone();

    int width = srcMat.cols;
    int height = srcMat.rows;
    int num_iters = 5;
    float time_opencv = 0;
    float time_dpcpp = 0;

    int *himage_dst, *dimage_dst;

    himage_dst =
        sycl::malloc_host<int>(width * height, dpct::get_default_queue());
    dimage_dst =
        sycl::malloc_host<int>(width * height, dpct::get_default_queue());

    for (int i = 0; i < num_iters; i++)
    {
        const auto t1 = get_time_point();
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Point2i erode_point(-1, -1);
        cv::morphologyEx(srcMat, dstMat, cv::MORPH_OPEN, element, erode_point, 1, BORDER_REPLICATE);
        const auto t2 = get_time_point();
        time_opencv += get_duration(t1, t2);
    }
    std::cout << "MorphOpen time consumption on CPU:" << time_opencv / num_iters << " ms" << std::endl;

    MorphOpenEsimd(srcMat.data, dstMat_tmp.data, dstMat_dpcpp.data, width, height, 1);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            himage_dst[i * width + j] = dstMat.data[i * width + j];
            dimage_dst[i * width + j] = dstMat_dpcpp.data[i * width + j];
        }
    }

    diff(himage_dst, dimage_dst, width, height);

    cv::cvtColor(dstMat, dstMat, COLOR_GRAY2BGR);
    cv::cvtColor(dstMat_dpcpp, dstMat_dpcpp, COLOR_GRAY2BGR);
    cv::imwrite("./data/dstMat.ppm", dstMat);
    cv::imwrite("./data/dstMat_dpcpp.ppm", dstMat_dpcpp);
    return 0;
}
