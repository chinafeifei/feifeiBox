//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
// pcb_opencv.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
// 本程序读取一系列传送带上的 pcb 电路板照片
// 经过 opencv resize，背景分离，threshold，闭运算，在众多图片中识别出 pcb 电路板在正中间的图片
//

#include <iostream>

#include <opencv2/opencv.hpp>

#include "ippcore_tl.h"
#include "ippi_tl.h"
#include "ipps.h"
#include "ippi.h"

#include <string>

#include "ipp_header.h"

#define IPP_MEAN      // 开启它，使用 ipp 版本的 boxfilter
#define IPP_RESIZE    // 开启它，使用 ipp 版本的 resize
#define IPP_THRESHOLD // 开启它，使用 ipp 版本的 threshold
#define IPP_MORPH     // 开启它，使用 ipp 版本的 morph

const bool DEBUG = false; // 开启 DEBUG ，可以看到每一个步骤的中间结果
const bool INFO = false;

void m_showimage(cv::Mat &image)
{

    if (DEBUG)
    {
        cv::imshow("image", image);
        cv::waitKey();
    }
}

int getMaxAreaContourId(std::vector<std::vector<cv::Point>> &contours)
{
    double maxArea = 0;
    int maxAreaContourId = -1;

    for (int i = 0; i < contours.size(); i++)
    {
        double newArea = cv::contourArea(contours.at(i));

        if (newArea > maxArea)
        {
            maxArea = newArea;
            maxAreaContourId = i;
        } // End if
    }     // End for
    return maxAreaContourId;
}

float check_cv(cv::Mat &color_image, cv::Mat &gray_image, cv::Ptr<cv::BackgroundSubtractor> &sub_MOG2)
{

    cv::Mat view_image;

    cv::resize(color_image, view_image, cv::Size(480, 270) /* , 0.25, 0.25 */);

    int n_total_px = 300000;

    int n_right_px = 1000;
    int n_left_px = 1000;
    int ratio = 4;

    // Total white pixel # on MOG applied
    // frame after morphological operations
    n_total_px = n_total_px / (ratio * ratio);

    n_left_px = n_left_px / (ratio * ratio);

    n_right_px = n_right_px / (ratio * ratio);

    int image_height = color_image.rows;
    int image_width = color_image.cols;

    cv::Mat resized_image;

    // 开始计时
    auto start = std::chrono::steady_clock::now();

    // 均值滤波
    cv::Mat mean_image;
    cv::boxFilter(gray_image, mean_image, gray_image.depth(), cv::Size(11, 11));
    m_showimage(mean_image);

    // 将图片缩小为原来的 25%

    auto resize_start = std::chrono::steady_clock::now();
    // 将图片缩小为原来的 1/16
    cv::resize(mean_image, resized_image, cv::Size(480, 270) /*, 0.25, 0.25 */);
    m_showimage(resized_image);
    auto resize_stop = std::chrono::steady_clock::now();

    auto mog_start = std::chrono::steady_clock::now();
    // 前景背景分离
    cv::Mat mog2;
    sub_MOG2->apply(resized_image, mog2);
    m_showimage(mog2);
    auto mog_stop = std::chrono::steady_clock::now();

    // printf("mog channel number is %d \n", mog2.channels());

    auto threshold_start = std::chrono::steady_clock::now();
    // 二值化
    cv::Mat threshold;
    cv::threshold(mog2, threshold, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    m_showimage(threshold);
    auto threshold_stop = std::chrono::steady_clock::now();

    auto close_start = std::chrono::steady_clock::now();
    // 闭运算
    cv::Mat morph;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
    cv::morphologyEx(threshold, morph, cv::MORPH_CLOSE, kernel);
    m_showimage(morph);
    auto close_stop = std::chrono::steady_clock::now();

    // 结束计时
    auto stop = std::chrono::steady_clock::now();
    auto ipp_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;

    if (INFO)
        printf("boxblur_resize_MOG_threshold_close %.3f milliseconds \n", ipp_time);
    // printf("resize                                  took %.3f milliseconds \n", resize_time);
    // printf("mog                                                     took %.3f milliseconds \n", mog_time);

    // 获取图片左边缘的 10 列元素，探查白色像素点的数量
    cv::Rect rect(0, 0, 10, morph.rows);
    cv::Mat image_cut_left = cv::Mat(morph, rect);
    // m_showimage(image_cut_left);
    int noZeroNum_left = cv::countNonZero(image_cut_left);

    if (DEBUG)
        printf("no zero pixel number is %d \n", noZeroNum_left);

    // 获取图片右边缘的 10 列元素，探查白色像素点的数量
    cv::Rect rect_r(morph.cols - 11, 0, 10, morph.rows);
    cv::Mat image_cut_right = cv::Mat(morph, rect_r);
    // m_showimage(image_cut_right);
    int noZeroNum_right = cv::countNonZero(image_cut_right);
    if (DEBUG)
        printf("no zero pixel number is %d \n", noZeroNum_right);

    // 获取整张图片的白色像素点的数量
    int noZeroTotal = cv::countNonZero(morph);
    if (DEBUG)
        printf("no zero pixel Total number is %d \n", noZeroTotal);

    // 如果图片左边和右边的白色像素足够少，但是图片的总白色像素足够多，我们认为目标在图片的正中间
    if (noZeroTotal > n_total_px && noZeroNum_left < n_left_px && noZeroNum_right < n_right_px)
    {

        if (DEBUG)
            printf("suhao           BINGO middle! \n");

        cv::Mat contour_image;
        cv::Rect bounding_rect;

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(morph, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

        std::vector<cv::Point> maxContour = contours.at(getMaxAreaContourId(contours));

        bounding_rect = cv::boundingRect(maxContour);

        cv::rectangle(view_image, bounding_rect.tl(), bounding_rect.br(), cv::Scalar(0, 255, 0), 2);

        m_showimage(view_image);
    }
    else
    {
        if (DEBUG)
            printf("not middle \n");
    }

    return ipp_time;
}

float check_ipp(cv::Mat &color_image, cv::Mat &gray_image, cv::Ptr<cv::BackgroundSubtractor> &sub_MOG2)
{

    cv::Mat view_image;

    cv::resize(color_image, view_image, cv::Size(480, 270) /* , 0.25, 0.25 */);

    int n_total_px = 300000;

    int n_right_px = 1000;
    int n_left_px = 1000;
    int ratio = 4;

    // Total white pixel # on MOG applied
    // frame after morphological operations
    n_total_px = n_total_px / (ratio * ratio);

    n_left_px = n_left_px / (ratio * ratio);

    n_right_px = n_right_px / (ratio * ratio);

    int image_height = color_image.rows;
    int image_width = color_image.cols;

    cv::Mat resized_image;

    // 开始计时
    auto start = std::chrono::steady_clock::now();

    // 均值滤波
    cv::Mat mean_image;
    tbb_ipp_boxfilter_11x11(gray_image, mean_image);
    m_showimage(mean_image);

    // 将图片缩小为原来的 25%

    auto resize_start = std::chrono::steady_clock::now();
    // 将图片缩小为原来的 1/16
    resizeIPP_Linear(mean_image, resized_image, cv::Size(480, 270));
    m_showimage(resized_image);
    auto resize_stop = std::chrono::steady_clock::now();

    auto mog_start = std::chrono::steady_clock::now();
    // 前景背景分离
    cv::Mat mog2;
    sub_MOG2->apply(resized_image, mog2);
    m_showimage(mog2);
    auto mog_stop = std::chrono::steady_clock::now();

    // printf("mog channel number is %d \n", mog2.channels());

    auto threshold_start = std::chrono::steady_clock::now();
    // 二值化
    cv::Mat threshold;
    Ipp8u otsuThreshold = 0;
    ippiComputeThreshold_Otsu_8u_C1R(mog2.data, mog2.step[0], IppiSize{mog2.cols, mog2.rows}, &otsuThreshold);
    ipp_threshold(mog2, threshold, otsuThreshold);
    m_showimage(threshold);
    auto threshold_stop = std::chrono::steady_clock::now();

    auto close_start = std::chrono::steady_clock::now();
    // 闭运算
    cv::Mat morph;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
    ipp_morph_close(threshold, morph, kernel);

    m_showimage(morph);
    auto close_stop = std::chrono::steady_clock::now();

    // 结束计时
    auto stop = std::chrono::steady_clock::now();
    auto ipp_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;

    if (INFO)
        printf("boxblur_resize_MOG_threshold_close %.3f milliseconds \n", ipp_time);
    // printf("resize                                  took %.3f milliseconds \n", resize_time);
    // printf("mog                                                     took %.3f milliseconds \n", mog_time);

    // 获取图片左边缘的 10 列元素，探查白色像素点的数量
    cv::Rect rect(0, 0, 10, morph.rows);
    cv::Mat image_cut_left = cv::Mat(morph, rect);
    // m_showimage(image_cut_left);
    int noZeroNum_left = cv::countNonZero(image_cut_left);

    if (DEBUG)
        printf("no zero pixel number is %d \n", noZeroNum_left);

    // 获取图片右边缘的 10 列元素，探查白色像素点的数量
    cv::Rect rect_r(morph.cols - 11, 0, 10, morph.rows);
    cv::Mat image_cut_right = cv::Mat(morph, rect_r);
    // m_showimage(image_cut_right);
    int noZeroNum_right = cv::countNonZero(image_cut_right);
    if (DEBUG)
        printf("no zero pixel number is %d \n", noZeroNum_right);

    // 获取整张图片的白色像素点的数量
    int noZeroTotal = cv::countNonZero(morph);
    if (DEBUG)
        printf("no zero pixel Total number is %d \n", noZeroTotal);

    // 如果图片左边和右边的白色像素足够少，但是图片的总白色像素足够多，我们认为目标在图片的正中间
    if (noZeroTotal > n_total_px && noZeroNum_left < n_left_px && noZeroNum_right < n_right_px)
    {

        if (DEBUG)
            printf("suhao           BINGO middle! \n");

        cv::Mat contour_image;
        cv::Rect bounding_rect;

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(morph, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

        std::vector<cv::Point> maxContour = contours.at(getMaxAreaContourId(contours));

        bounding_rect = cv::boundingRect(maxContour);

        cv::rectangle(view_image, bounding_rect.tl(), bounding_rect.br(), cv::Scalar(0, 255, 0), 2);

        m_showimage(view_image);
    }
    else
    {
        if (DEBUG)
            printf("not middle \n");
    }

    return ipp_time;
}

float check_debug(cv::Mat &color_image, cv::Mat &gray_image, cv::Ptr<cv::BackgroundSubtractor> &sub_MOG2)
{

    cv::Mat view_image;

    cv::resize(color_image, view_image, cv::Size(480, 270) /* , 0.25, 0.25 */);

    int n_total_px = 300000;

    int n_right_px = 1000;
    int n_left_px = 1000;
    int ratio = 4;

    // Total white pixel # on MOG applied
    // frame after morphological operations
    n_total_px = n_total_px / (ratio * ratio);

    n_left_px = n_left_px / (ratio * ratio);

    n_right_px = n_right_px / (ratio * ratio);

    int image_height = color_image.rows;
    int image_width = color_image.cols;

    cv::Mat resized_image;

    // 开始计时
    auto start = std::chrono::steady_clock::now();

    // 均值滤波
    cv::Mat mean_image;
#ifdef IPP_MEAN
    tbb_ipp_boxfilter_11x11(gray_image, mean_image);
#else
    cv::boxFilter(gray_image, mean_image, gray_image.depth(), cv::Size(11, 11));
#endif
    m_showimage(mean_image);

    // 将图片缩小为原来的 25%

    auto resize_start = std::chrono::steady_clock::now();
    // 将图片缩小为原来的 1/16
#ifdef IPP_RESIZE
    // resize_ipp(mean_image, resized_image, cv::Size(480, 270) /*, 0.25, 0.25*/ );
    resizeIPP_Linear(mean_image, resized_image, cv::Size(480, 270));
    // resizeIPP_Nearest_C3R(image, resized_image, cv::Size(), 0.25, 0.25);
    // resizeIPP_Linear_C3R(image, resized_image, cv::Size(), 0.25, 0.25);
    // resize_ipp_tl(image, resized_image, cv::Size(), 0.25, 0.25);
#else
    cv::resize(mean_image, resized_image, cv::Size(480, 270) /*, 0.25, 0.25 */);
#endif
    m_showimage(resized_image);
    auto resize_stop = std::chrono::steady_clock::now();

    auto mog_start = std::chrono::steady_clock::now();
    // 前景背景分离
    cv::Mat mog2;
    sub_MOG2->apply(resized_image, mog2);
    m_showimage(mog2);
    auto mog_stop = std::chrono::steady_clock::now();

    // printf("mog channel number is %d \n", mog2.channels());

    auto threshold_start = std::chrono::steady_clock::now();
    // 二值化
    cv::Mat threshold;
#ifdef IPP_THRESHOLD

    Ipp8u otsuThreshold = 0;
    ippiComputeThreshold_Otsu_8u_C1R(mog2.data, mog2.step[0], IppiSize{mog2.cols, mog2.rows}, &otsuThreshold);
    ipp_threshold(mog2, threshold, otsuThreshold);
    // ipp_threshold_tbb(mog2, threshold, otsuThreshold);
    // ipp_threshold_omp(mog2, threshold, otsuThreshold);

#else
    cv::threshold(mog2, threshold, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
#endif
    m_showimage(threshold);
    auto threshold_stop = std::chrono::steady_clock::now();

    auto close_start = std::chrono::steady_clock::now();
    // 闭运算
    cv::Mat morph;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
#ifdef IPP_MORPH
    // ipp_morph_close_tbb(threshold, morph, kernel);
    ipp_morph_close(threshold, morph, kernel);
#else
    cv::morphologyEx(threshold, morph, cv::MORPH_CLOSE, kernel);
#endif
    m_showimage(morph);
    auto close_stop = std::chrono::steady_clock::now();

    // 结束计时
    auto stop = std::chrono::steady_clock::now();
    auto ipp_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;

    if (INFO)
        printf("boxblur_resize_MOG_threshold_close %.3f milliseconds \n", ipp_time);
    // printf("resize                                  took %.3f milliseconds \n", resize_time);
    // printf("mog                                                     took %.3f milliseconds \n", mog_time);

    // 获取图片左边缘的 10 列元素，探查白色像素点的数量
    cv::Rect rect(0, 0, 10, morph.rows);
    cv::Mat image_cut_left = cv::Mat(morph, rect);
    // m_showimage(image_cut_left);
    int noZeroNum_left = cv::countNonZero(image_cut_left);

    if (DEBUG)
        printf("no zero pixel number is %d \n", noZeroNum_left);

    // 获取图片右边缘的 10 列元素，探查白色像素点的数量
    cv::Rect rect_r(morph.cols - 11, 0, 10, morph.rows);
    cv::Mat image_cut_right = cv::Mat(morph, rect_r);
    // m_showimage(image_cut_right);
    int noZeroNum_right = cv::countNonZero(image_cut_right);
    if (DEBUG)
        printf("no zero pixel number is %d \n", noZeroNum_right);

    // 获取整张图片的白色像素点的数量
    int noZeroTotal = cv::countNonZero(morph);
    if (DEBUG)
        printf("no zero pixel Total number is %d \n", noZeroTotal);

    // 如果图片左边和右边的白色像素足够少，但是图片的总白色像素足够多，我们认为目标在图片的正中间
    if (noZeroTotal > n_total_px && noZeroNum_left < n_left_px && noZeroNum_right < n_right_px)
    {

        if (DEBUG)
            printf("suhao           BINGO middle! \n");

        cv::Mat contour_image;
        cv::Rect bounding_rect;

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(morph, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

        std::vector<cv::Point> maxContour = contours.at(getMaxAreaContourId(contours));

        bounding_rect = cv::boundingRect(maxContour);

        cv::rectangle(view_image, bounding_rect.tl(), bounding_rect.br(), cv::Scalar(0, 255, 0), 2);

        m_showimage(view_image);
    }
    else
    {
        if (DEBUG)
            printf("not middle \n");
    }

    return ipp_time;
}

// int main_pcb_opencv_02_ipp()
int main()
{

    printf("\nPCBA Preprocessing\n");

    cv::Ptr<cv::BackgroundSubtractor> sub_MOG2 = cv::createBackgroundSubtractorMOG2();

    float total_ipp_time = 0.0f;
    float total_cv_time = 0.0f;

    int LOOP_NUM = 27;

    // HANDLE mHandle = GetCurrentProcess();
    // BOOL result = SetPriorityClass(mHandle, REALTIME_PRIORITY_CLASS);
    // SetThreadAffinityMask(GetCurrentThread(), 0x00000001); // 0x01，Core0

    // std::string filename = "data/1k.jpg";
    // cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

    for (int i = 0; i < LOOP_NUM; i++)
    {

        std::string path = "video/";

        std::string all_path = path + std::to_string(i) + ".jpg";

        std::string filename = "data/1k.jpg";

        cv::Mat image = cv::imread(all_path, cv::IMREAD_COLOR);

        if (DEBUG)
            std::cout << all_path << std::endl;

        cv::Mat color_image = cv::imread(all_path, cv::IMREAD_COLOR);
        cv::Mat gray_image = cv::imread(all_path, cv::IMREAD_GRAYSCALE);

        // cv::resize(gray_image, gray_image, cv::Size(2560, 1440) );
        // cv::resize(gray_image, gray_image, cv::Size(3840, 2160) );

        // cv::imshow("image", image);
        // cv::waitKey();

        if (i == 0)
        {
            check_ipp(color_image, gray_image, sub_MOG2);
            check_cv(color_image, gray_image, sub_MOG2);
            continue;
        }

        total_ipp_time += check_ipp(color_image, gray_image, sub_MOG2);
        total_cv_time += check_cv(color_image, gray_image, sub_MOG2);
    }

    float average_cv = total_cv_time / (LOOP_NUM - 1);
    float average_ipp = total_ipp_time / (LOOP_NUM - 1);

    printf("\nAverage opencv time is %.3f millisecond\n", average_cv);

    printf("\nAverage ipp    time is %.3f millisecond\n", average_ipp);

    printf("\nIpp speeds up %.1f %% \n", average_cv / average_ipp * 100);
}
