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
#include "sobel.h"

void diff(float *himage, float *dimage, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (abs(himage[i * width + j] - dimage[i * width + j]) > 1)
            {
                std::cout << "Expected: " << himage[i * width + j] << ", actual: " << dimage[i * width + j] << ", on: " << i << ", " << j << std::endl;
                exit(0);
            }
        }
    }
}

void sobel_opencv_sum(cv::Mat &src, float scale, cv::Mat &dst)
{

    int ksize = 3;
    float delta = 0;
    float elapsed = 0.0;

    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    int num_iters = 5;

    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < num_iters; i++)
    {
        cv::Sobel(src, grad_x, CV_32F, 1, 0, ksize, scale, delta, cv::BORDER_REPLICATE);
        cv::Sobel(src, grad_y, CV_32F, 0, 1, ksize, scale, delta, cv::BORDER_REPLICATE);

        abs_grad_x = cv::abs(grad_x);
        abs_grad_y = cv::abs(grad_y);

        cv::add(abs_grad_x, abs_grad_y, dst);
    }
    auto end_time = std::chrono::steady_clock::now();

    elapsed = std::chrono::duration<float, std::milli>(end_time - start_time)
                  .count();

    std::cout << "The sobel sum elapsed time in opencv was " << elapsed / num_iters << " ms" << std::endl;
}

int main(int argc, char **argv)
{

    float *himage_dst, *dimage_dst;

    cv::Mat srcMat = cv::imread("data/color_4288.jpg", 0);
    if (srcMat.empty())
    {
        printf("srcMat is empty\n");
        return -1;
    }

    cv::Mat dstMat = srcMat.clone();
    cv::Mat dstMat_dpcpp = srcMat.clone();
    int thd_per_blk = 256;

    int width = srcMat.cols;
    int height = srcMat.rows;
    float imageScale = 1.f;

    himage_dst = (float *)malloc(width * height * sizeof(float));
    dimage_dst = (float *)malloc(width * height * sizeof(float));

    // opencv sobel
    sobel_opencv_sum(srcMat, imageScale, dstMat);

    // dpcpp sobel
    apply_sobel_sum(dimage_dst, srcMat.data, imageScale, width, height, thd_per_blk);

    dstMat_dpcpp.convertTo(dstMat_dpcpp, CV_32F);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            himage_dst[i * width + j] = dstMat.at<float>(i, j);
            dstMat_dpcpp.at<float>(i, j) = dimage_dst[i * width + j];
        }
    }

    diff(himage_dst, dimage_dst, width, height);

    std::cout << "Great!!" << std::endl;

    cv::cvtColor(dstMat, dstMat, cv::COLOR_GRAY2BGR);
    cv::cvtColor(dstMat_dpcpp, dstMat_dpcpp, cv::COLOR_GRAY2BGR);
    free(himage_dst);
    free(dimage_dst);
    cv::imwrite("./data/dstMat_sum.ppm", dstMat);
    cv::imwrite("./data/dstMat_dpcpp_sum.ppm", dstMat_dpcpp);

    return 0;
}
