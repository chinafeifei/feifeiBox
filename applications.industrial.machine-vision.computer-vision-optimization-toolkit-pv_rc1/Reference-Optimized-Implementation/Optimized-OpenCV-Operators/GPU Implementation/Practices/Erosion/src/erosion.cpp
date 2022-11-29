//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <string.h>
#include <stdlib.h>
//#include <oneapi/mkl.hpp>
//#include <oneapi/mkl/rng/device.hpp>

#include <chrono>
#include <ctime>

// #include "erosionFuncTemplate.h"
// #include "erosionCPU.h"
#include "erosion.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;

using time_point = decltype(std::chrono::steady_clock::now());
static inline time_point get_time_point()
{
    waitForKernelCompletion();
    return std::chrono::steady_clock::now();
}

static inline double get_duration(const time_point &from, const time_point &to)
{
    ;
    return std::chrono::duration<float, std::milli>(to - from).count();
}

inline int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    deviceCount = dpct::dev_mgr::instance().device_count();

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    dpct::dev_mgr::instance().select_device(0);

    return 0;
}

void populateImage(int *image, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            image[i * width + j] = rand() % 256;
        }
    }
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

int main(int argc, char *argv[])
{
    cudaDeviceInit(argc, (const char **)argv);

    int *himage_dst, *dimage_dst;
    // Width and height of the image
    cv::Mat srcMat = imread("./data/color_4288.jpg", 0);
    if (srcMat.empty())
    {
        printf("srcMat is empty\n");
        return -1;
    }
    cv::Mat dstMat = srcMat.clone();
    cv::Mat dstMat_dpcpp = srcMat.clone();

    // Property of srcMat
    int width = srcMat.cols;
    int height = srcMat.rows;
    int ratio = 2;
    int num_iters = 5;
    float time_opencv = 0;
    float time_dpcpp = 0;

    himage_dst =
        sycl::malloc_host<int>(width * height, dpct::get_default_queue());
    /*
    DPCT1003:19: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    dimage_dst =
        sycl::malloc_host<int>(width * height, dpct::get_default_queue());

    // Randomly populate the image
    // populateImage(himage_src, width, height);

    for (int i = 0; i < num_iters; i++)
    {
        auto start = get_time_point();

        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * ratio + 1, 2 * ratio + 1));
        cv::Point2i erode_point(-1, -1);
        cv::erode(srcMat, dstMat, element, erode_point, 1, BORDER_REPLICATE);

        auto end = get_time_point();
        time_opencv += get_duration(start, end);
    }
    std::cout << "Erosion CPU: " << time_opencv / num_iters << "ms\n";

    // Erosion with kernel size 3*3
    // ErosionEsimd(srcMat.data, dstMat_dpcpp.data, width, height, ratio);
    // Erosion with kernel size 5*5
    ErosionEsimd5x5(srcMat.data, dstMat_dpcpp.data, width, height, ratio);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            himage_dst[i * width + j] = dstMat.data[i * width + j];
            dimage_dst[i * width + j] = dstMat_dpcpp.data[i * width + j];
        }
    }

    diff(himage_dst, dimage_dst, width, height);

    std::cout << "Great!!" << std::endl;
    sycl::free(himage_dst, dpct::get_default_queue());
    sycl::free(dimage_dst, dpct::get_default_queue());
    dpct::get_current_device().reset();

    cvtColor(dstMat, dstMat, COLOR_GRAY2BGR);
    cvtColor(dstMat_dpcpp, dstMat_dpcpp, COLOR_GRAY2BGR);
    cv::imwrite("./data/dstMat.ppm", dstMat);
    cv::imwrite("./data/dstMat_dpcpp.ppm", dstMat_dpcpp);

    return 0;
}
