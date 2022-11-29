//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <deque>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "canny.h"

// const char* CW_IMG_ORIGINAL = "Original";
// const char* CW_IMG_GRAY = "Grayscale";
// const char* CW_IMG_EDGE = "Canny Edge Detection";
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

void edge_hysteresis(uint8_t *dimage, uint8_t *dimage_week, int width, int height)
{
    std::deque<int> edge_stack;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (dimage[i * width + j] == 255)
            {
                edge_stack.push_back(i * width + j);
            }
        }
    }

    while (!edge_stack.empty())
    {
        int y = edge_stack.back() / width;
        int x = edge_stack.back() % width;
        edge_stack.pop_back();

        for (int i = -1; i < 2; i++)
        {
            for (int j = -1; j < 2; j++)
            {
                if ((y + i) < 0 || (y + i) >= height || (x + j) < 0 || (x + j) >= width)
                    continue;
                if (dimage_week[(y + i) * width + (x + j)] == 1)
                {
                    edge_stack.push_back((y + i) * width + (x + j));
                    dimage[(y + i) * width + (x + j)] = 255;
                    dimage_week[(y + i) * width + (x + j)] = 0;
                }
            }
        }
    }
}

int main(int argc, char **argv)
{

    int *dimage_dst, *himage_dst;
    uint8_t *dimage_week;

    cv::Mat srcMat = cv::imread("./data/color_4288.jpg", 0);
    if (srcMat.empty())
    {
        printf("srcMat is empty\n");
        return -1;
    }

    cv::Mat dstMat = srcMat.clone();
    cv::Mat dstMat_dpcpp = srcMat.clone();
    cv::Mat dx, dy;

    int width = srcMat.cols;
    int height = srcMat.rows;
    int thd_per_blk = 256;
    int low_threshold = 30;
    int high_threshold = 90;

    himage_dst = (int *)malloc(width * height * sizeof(int));
    dimage_dst = (int *)malloc(width * height * sizeof(int));
    dimage_week = (uint8_t *)malloc(width * height * sizeof(uint8_t));

    int num_iters = 5;
    float time_opencv = 0;
    float time_dpcpp = 0;

    for (int i = 0; i < num_iters; i++)
    {
        const auto t1 = std::chrono::steady_clock::now();
        cv::GaussianBlur(srcMat, dstMat, cv::Size(3, 3), 0, 0, cv::BORDER_REPLICATE);
        cv::Sobel(dstMat, dx, CV_16S, 1, 0, 3, 1.0, 0, cv::BORDER_REPLICATE);
        cv::Sobel(dstMat, dy, CV_16S, 0, 1, 3, 1.0, 0, cv::BORDER_REPLICATE);
        cv::Canny(dx, dy, dstMat, low_threshold, high_threshold, true);
        const auto t2 = std::chrono::steady_clock::now();
        time_opencv += std::chrono::duration<float, std::milli>(t2 - t1)
                           .count();
    }

    std::cout << "the time of opencv canny is :" << time_opencv / num_iters << " ms" << std::endl;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            dimage_week[i * width + j] = 0;
        }
    }

    apply_canny(dstMat_dpcpp.data, dimage_week, srcMat.data, low_threshold, high_threshold, width, height, thd_per_blk);
    for (int i = 0; i < num_iters; i++)
    {
        const auto t3 = std::chrono::steady_clock::now();
        edge_hysteresis(dstMat_dpcpp.data, dimage_week, width, height);
        const auto t4 = std::chrono::steady_clock::now();
        time_dpcpp += std::chrono::duration<float, std::milli>(t4 - t3)
                          .count();
    }

    std::cout << "The elapsed time of dpcpp-canny in cpu steps was :" << time_dpcpp / num_iters << " ms" << std::endl;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            // dstMat_dpcpp.data[i * width + j] = dimage_dst[i * width + j];
            himage_dst[i * width + j] = dstMat.data[i * width + j];
            dimage_dst[i * width + j] = dstMat_dpcpp.data[i * width + j];
        }
    }

    diff(himage_dst, dimage_dst, width, height);

    std::cout << "Great!!" << std::endl;

    cv::cvtColor(dstMat, dstMat, cv::COLOR_GRAY2BGR);
    cv::cvtColor(dstMat_dpcpp, dstMat_dpcpp, cv::COLOR_GRAY2BGR);
    free(himage_dst);
    free(dimage_dst);
    free(dimage_week);
    cv::imwrite("./data/dstMat.ppm", dstMat);
    cv::imwrite("./data/dstMat_dpcpp.ppm", dstMat_dpcpp);

    return 0;
}
