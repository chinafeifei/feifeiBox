//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/imgproc.hpp>

#include <opencv2/highgui.hpp>

#include "util.h"

#include "ippi.h"
#include "ippi_tl.h"
#include "ipps.h"

const bool DEBUG = false;

int resizeIPP_Linear(cv::Mat &image, cv::Mat &dst, cv::Size dsize,
                     double inv_scale_x = 0, double inv_scale_y = 0)
{

    cv::Size ssize = image.size();

    if (dsize.empty())
    {

        dsize = cv::Size(cv::saturate_cast<int>(ssize.width * inv_scale_x),
                         cv::saturate_cast<int>(ssize.height * inv_scale_y));
    }

    auto create_start = std::chrono::high_resolution_clock::now();
    dst.create(dsize, image.type());
    auto create_stop = std::chrono::high_resolution_clock::now();
    auto create_time = std::chrono::duration_cast<std::chrono::microseconds>(create_stop - create_start).count() / 1000.0;
    // printf("ipp linear resize create mat time is %.3f milliseconds \n", create_time );

    if (dsize == ssize)
    {
        image.copyTo(dst);
        return;
    }

    Ipp8u *pSrc = (Ipp8u *)image.data;
    IppiSize srcSize = {image.cols, image.rows};
    Ipp32s srcStep = image.step[0];
    Ipp8u *pDst = (Ipp8u *)dst.data;
    IppiSize dstSize = {dsize.width, dsize.height};
    Ipp32s dstStep = dst.step[0];

    IppiResizeSpec_32f *pSpec = 0;
    int specSize = 0, initSize = 0, bufSize = 0;
    Ipp8u *pBuffer = 0;
    Ipp8u *pInitBuf = 0;
    Ipp32u numChannels = image.channels();
    IppiPoint dstOffset = {0, 0};
    IppStatus status = ippStsNoErr;
    IppiBorderType border = ippBorderRepl;

    /* Spec and init buffer sizes 获取 spec 大小，*/
    status = ippiResizeGetSize_8u(srcSize, dstSize, ippLinear, 0, &specSize, &initSize);

    if (status != ippStsNoErr)
        return status;

    /* Memory allocation 创建 spec 内存 ，初始化缓存 */
    pInitBuf = ippsMalloc_8u(initSize);
    pSpec = (IppiResizeSpec_32f *)ippsMalloc_8u(specSize);

    // 错误控制
    if (pInitBuf == NULL || pSpec == NULL)
    {
        ippsFree(pInitBuf);
        ippsFree(pSpec);
        return ippStsNoMemErr;
    }

    /* Filter initialization 初始化 sSpec  */
    status = ippiResizeLinearInit_8u(srcSize, dstSize, pSpec);
    ippsFree(pInitBuf);

    // 错误控制
    if (status != ippStsNoErr)
    {
        ippsFree(pSpec);
        return status;
    }

    /* work buffer size 计算工作缓存的大小 */
    status = ippiResizeGetBufferSize_8u(pSpec, dstSize, numChannels, &bufSize);
    if (status != ippStsNoErr)
    {
        ippsFree(pSpec);
        return status;
    }

    // 创建工作缓存
    pBuffer = ippsMalloc_8u(bufSize);
    if (pBuffer == NULL)
    {
        ippsFree(pSpec);
        return ippStsNoMemErr;
    }

    /* Resize processing */
    if (numChannels == 3)
    {
        status = ippiResizeLinear_8u_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, ippBorderRepl, 0, pSpec, pBuffer);
    }
    else
    {
        status = ippiResizeLinear_8u_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, ippBorderRepl, 0, pSpec, pBuffer);
    }

    // 销毁 spec 和 工作缓存
    ippsFree(pSpec);
    ippsFree(pBuffer);
    return status;
}

int resizeIPP_Nearest_C3R(cv::Mat &image, cv::Mat &dst, cv::Size dsize,
                          double inv_scale_x, double inv_scale_y)
{

    cv::Size ssize = image.size();

    if (dsize.empty())
    {

        dsize = cv::Size(cv::saturate_cast<int>(ssize.width * inv_scale_x),
                         cv::saturate_cast<int>(ssize.height * inv_scale_y));
    }

    dst.create(dsize, image.type());

    if (dsize == ssize)
    {
        image.copyTo(dst);
        return;
    }

    Ipp8u *pSrc = (Ipp8u *)image.data;
    IppiSize srcSize = {image.cols, image.rows};
    Ipp32s srcStep = image.step[0];
    Ipp8u *pDst = (Ipp8u *)dst.data;
    IppiSize dstSize = {dsize.width, dsize.height};
    Ipp32s dstStep = dst.step[0];

    IppiResizeSpec_32f *pSpec = 0;
    int specSize = 0, initSize = 0, bufSize = 0;
    Ipp8u *pBuffer = 0;
    Ipp8u *pInitBuf = 0;
    Ipp32u numChannels = 3;
    IppiPoint dstOffset = {0, 0};
    IppStatus status = ippStsNoErr;
    IppiBorderType border = ippBorderRepl;

    /* Spec and init buffer sizes 获取 spec 大小，*/
    status = ippiResizeGetSize_8u(srcSize, dstSize, ippNearest, 0, &specSize, &initSize);

    if (status != ippStsNoErr)
        return status;

    /* Memory allocation 创建 spec 内存 ，初始化缓存 */
    pInitBuf = ippsMalloc_8u(initSize);
    pSpec = (IppiResizeSpec_32f *)ippsMalloc_8u(specSize);

    // 错误控制
    if (pInitBuf == NULL || pSpec == NULL)
    {
        ippsFree(pInitBuf);
        ippsFree(pSpec);
        return ippStsNoMemErr;
    }

    /* Filter initialization 初始化 sSpec  */
    status = ippiResizeNearestInit_8u(srcSize, dstSize, pSpec);
    ippsFree(pInitBuf);

    // 错误控制
    if (status != ippStsNoErr)
    {
        ippsFree(pSpec);
        return status;
    }

    /* work buffer size 计算工作缓存的大小 */
    status = ippiResizeGetBufferSize_8u(pSpec, dstSize, numChannels, &bufSize);
    if (status != ippStsNoErr)
    {
        ippsFree(pSpec);
        return status;
    }

    // 创建工作缓存
    pBuffer = ippsMalloc_8u(bufSize);
    if (pBuffer == NULL)
    {
        ippsFree(pSpec);
        return ippStsNoMemErr;
    }

    /* Resize processing */
    status = ippiResizeNearest_8u_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);

    // 销毁 spec 和 工作缓存
    ippsFree(pSpec);
    ippsFree(pBuffer);
    return status;
}

IppStatus resizeIPP_Lanczos_C3R(cv::Mat &image, cv::Mat &dst, cv::Size dsize,
                                double inv_scale_x, double inv_scale_y)
{

    cv::Size ssize = image.size();

    if (dsize.empty())
    {

        dsize = cv::Size(cv::saturate_cast<int>(ssize.width * inv_scale_x),
                         cv::saturate_cast<int>(ssize.height * inv_scale_y));
    }

    dst.create(dsize, image.type());

    if (dsize == ssize)
    {
        image.copyTo(dst);
        return;
    }

    Ipp8u *pSrc = (Ipp8u *)image.data;
    IppiSize srcSize = {image.cols, image.rows};
    Ipp32s srcStep = image.step[0];
    Ipp8u *pDst = (Ipp8u *)dst.data;
    IppiSize dstSize = {dsize.width, dsize.height};
    Ipp32s dstStep = dst.step[0];

    IppiResizeSpec_32f *pSpec = 0;
    int specSize = 0, initSize = 0, bufSize = 0;
    Ipp8u *pBuffer = 0;
    Ipp8u *pInitBuf = 0;
    Ipp32u numChannels = 3;
    IppiPoint dstOffset = {0, 0};
    IppStatus status = ippStsNoErr;
    IppiBorderType border = ippBorderRepl;

    /* Spec and init buffer sizes 获取 spec 大小，*/
    status = ippiResizeGetSize_8u(srcSize, dstSize, ippLanczos, 0, &specSize, &initSize);

    if (status != ippStsNoErr)
        return status;

    /* Memory allocation 创建 spec 内存 ，初始化缓存 */
    pInitBuf = ippsMalloc_8u(initSize);
    pSpec = (IppiResizeSpec_32f *)ippsMalloc_8u(specSize);

    // 错误控制
    if (pInitBuf == NULL || pSpec == NULL)
    {
        ippsFree(pInitBuf);
        ippsFree(pSpec);
        return ippStsNoMemErr;
    }

    /* Filter initialization 初始化 sSpec  */
    status = ippiResizeLanczosInit_8u(srcSize, dstSize, 3, pSpec, pInitBuf);
    ippsFree(pInitBuf);

    // 错误控制
    if (status != ippStsNoErr)
    {
        ippsFree(pSpec);
        return status;
    }

    /* work buffer size 计算工作缓存的大小 */
    status = ippiResizeGetBufferSize_8u(pSpec, dstSize, numChannels, &bufSize);
    if (status != ippStsNoErr)
    {
        ippsFree(pSpec);
        return status;
    }

    // 创建工作缓存
    pBuffer = ippsMalloc_8u(bufSize);
    if (pBuffer == NULL)
    {
        ippsFree(pSpec);
        return ippStsNoMemErr;
    }

    /* Resize processing */
    status = ippiResizeLanczos_8u_C3R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, border, 0, pSpec, pBuffer);

    // 销毁 spec 和 工作缓存
    ippsFree(pSpec);
    ippsFree(pBuffer);
    return status;
}

void resize_ipp_tl(cv::Mat &src, cv::Mat &dst, cv::Size dsize,
                   double inv_scale_x, double inv_scale_y)
{
    cv::Size ssize = src.size();

    if (dsize.empty())
    {

        dsize = cv::Size(cv::saturate_cast<int>(ssize.width * inv_scale_x),
                         cv::saturate_cast<int>(ssize.height * inv_scale_y));
    }

    dst.create(dsize, src.type());

    if (dsize == ssize)
    {
        src.copyTo(dst);
        return;
    }

    IppiSizeL srcSize_TL, dstSize_TL;
    srcSize_TL.width = src.cols;
    srcSize_TL.height = src.rows;
    dstSize_TL.width = dsize.width;
    dstSize_TL.height = dsize.height;

    Ipp32s srcStep = src.step;
    Ipp32s dstStep = dst.step;

    Ipp8u *pSrc = (Ipp8u *)&src.data[0];
    Ipp8u *pDst = (Ipp8u *)&dst.data[0];

    IppSizeL specSize = 0;
    IppSizeL tempSize = 0;
    Ipp32u numChannels = src.channels();
    IppStatus ippStatus;
    IppiResizeSpec_LT *g_pSpec_TL = 0;

    ippStatus = ippiResizeGetSize_LT(srcSize_TL, dstSize_TL, ipp8u, ippLinear, 0, &specSize, &tempSize);
    g_pSpec_TL = (IppiResizeSpec_LT *)ippsMalloc_8u(specSize);
    ippStatus = ippiResizeLinearInit_LT(srcSize_TL, dstSize_TL, ipp8u, numChannels, g_pSpec_TL);
    // ippStatus = ippiResizeNearestInit_LT(srcSize_TL, dstSize_TL, ipp8u, numChannels, g_pSpec_TL);
    ippStatus = ippiResizeGetBufferSize_LT(g_pSpec_TL, &tempSize);
    Ipp8u *pBuffer = ippsMalloc_8u(tempSize);

    if (numChannels == 3)
        ippStatus = ippiResizeLinear_8u_C3R_LT(pSrc, srcStep, pDst, dstStep, ippBorderRepl, NULL, g_pSpec_TL, pBuffer);
    // ippStatus = ippiResizeNearest_8u_C3R_LT(pSrc, srcStep, pDst, dstStep, g_pSpec_TL, pBuffer);
    else
        ippStatus = ippiResizeLinear_8u_C1R_LT(pSrc, srcStep, pDst, dstStep, ippBorderRepl, NULL, g_pSpec_TL, pBuffer);
    // ippStatus = ippiResizeNearest_8u_C1R_LT(pSrc, srcStep, pDst, dstStep, g_pSpec_TL, pBuffer);

    return;
}

/**
 * 关于 ipp 程序的性能测试注意以下几点
 * 1. 在 for 循环内部读取输入图片，这样做更加贴近实际使用场景，同时也增加了 cache miss
 * 2. opencv 程序与 ipp 程序读取的输入图片不能是同一个 cv::Mat，这会增加 cache 命中率，干扰性能测试
 *
 */
int main_ipp_resize(int argc, char **argv)
// int main(int argc, char** argv)
{
    // std::string filename = "data/so_small_starry_25.png";
    // std::string filename = "data/color_4288.jpg";
    // std::string filename = "data/example_gray.bmp";

    const int LOOP_NUM = 200;
    const int WARM_NUM = 5;

    float average_cv = 0.0f;
    float average_ipp = 0.0f;
    float average_ipp_tl = 0.0f;
    for (int i = 0; i < LOOP_NUM; i++)
    {

        cv::Mat output;
        cv::Mat ippOutput;
        cv::Mat ippTLoutput;
        std::string filename = "data/1k.jpg";
        cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
        cv::Mat image_ipp = cv::imread(filename, cv::IMREAD_COLOR);
        cv::Mat image_ipp_tl = cv::imread(filename, cv::IMREAD_COLOR);

        auto cv_start = std::chrono::steady_clock::now();
        // cv::resize(image, output, cv::Size(), 0.25, 0.25, cv::INTER_NEAREST);
        cv::resize(image, output, cv::Size(), 0.25, 0.25, cv::INTER_LINEAR);
        auto cv_stop = std::chrono::steady_clock::now();
        auto cv_time = std::chrono::duration_cast<std::chrono::microseconds>(cv_stop - cv_start).count() / 1000.0;

        auto ipp_start = std::chrono::steady_clock::now();
        // resizeIPP_Nearest_C3R(image, ippOutput, cv::Size(), 0.25, 0.25);
        resizeIPP_Linear(image_ipp, ippOutput, cv::Size(), 0.25, 0.25);
        auto ipp_stop = std::chrono::steady_clock::now();
        auto ipp_time = std::chrono::duration_cast<std::chrono::microseconds>(ipp_stop - ipp_start).count() / 1000.0;

        auto ipp_lt_start = std::chrono::steady_clock::now();
        resize_ipp_tl(image_ipp_tl, ippTLoutput, cv::Size(), 0.25, 0.25);
        auto ipp_lt_stop = std::chrono::steady_clock::now();
        auto ipp_lt_time = std::chrono::duration_cast<std::chrono::microseconds>(ipp_lt_stop - ipp_lt_start).count() / 1000.0;

        if (i >= WARM_NUM)
        {
            average_cv += cv_time;
            average_ipp += ipp_time;
            average_ipp_tl += ipp_lt_time;
        }
        // printf("opencv time is %3.3f milliseconds\n", time);
    }
    printf("OPENCV average time is %.3f milliseconds \n", average_cv / (LOOP_NUM - WARM_NUM));
    printf("IPP    average time is %.3f milliseconds \n", average_ipp / (LOOP_NUM - WARM_NUM));
    printf("IPP TL average time is %.3f milliseconds \n", average_ipp_tl / (LOOP_NUM - WARM_NUM));

    printf("opencv result is \n");
    // displayArray(output, DEBUG);

    // float average = 0.0f;
    // for (int i = 0; i < LOOP_NUM; i++) {

    //    auto start = std::chrono::steady_clock::now();
    //    //resizeIPP_Nearest_C3R(image, ippOutput, cv::Size(), 0.25, 0.25);
    //    resizeIPP_Linear_C3R(image, ippOutput, cv::Size(), 0.25, 0.25);
    //    auto stop = std::chrono::steady_clock::now();
    //    auto time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;
    //    if (i >= WARM_NUM) {
    //        average += time;
    //    }
    //    //printf("ipp time is %.3f milliseconds\n", time);
    //}
    // printf("IPP average time is %.3f milliseconds \n", average / (LOOP_NUM - WARM_NUM) );

    // printf("ipp tl resize output is\n");
    // displayArray(ippOutput, DEBUG);

    // cv::imshow("image", ippOutput);
    // cv::waitKey();
}