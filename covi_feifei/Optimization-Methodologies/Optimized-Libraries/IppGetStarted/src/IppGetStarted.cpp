//==============================================================
/* Copyright(C) 2022 Intel Corporation
* Licensed under the Intel Proprietary License
*/
// =============================================================

#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "iw++/iw.hpp"
#include "ippi.h"

void compare_result(Ipp8u pSrc[], cv::Mat& smoothed)
{
    Ipp8u* pDst = smoothed.data;
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

    std::cout << "All pixels compared with OpenCV result : "
        << correctness
        << "% matched \n"
        << std::endl;
}

int main(void)
{
    /* Add funciton implement with OpenCV API */
    std::string img_path = "data/w08k.jpg";
    cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    cv::Mat dst_opencv;

    for (int i = 0; i < 100; i++) /* warm up */
    {
        cv::add(img, img, dst_opencv);
    }

    auto start_cv = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++)
    {
        cv::add(img, img, dst_opencv);
    }
    auto end_cv = std::chrono::high_resolution_clock::now();

    std::cout << "OpenCV Runtime : "
        << std::chrono::duration_cast<std::chrono::milliseconds>((end_cv - start_cv) / 100).count()
        << " ms\n"
        << std::endl;

    /* Add funciton implement with IW IPP API */
    ipp::IwiImage img_iw_ipp; ipp::IwiImage dst_iw_ipp;
    img_iw_ipp.Init(ipp::IwiSize(img.cols, img.rows), ipp8u, img.channels(), NULL, &img.data[0], img.cols * img.channels());
    dst_iw_ipp.Alloc(img_iw_ipp.m_size, ipp8u, img.channels());

    for (int i = 0; i < 100; i++) /* warm up */
    {
        ipp::iwiAdd(img_iw_ipp, img_iw_ipp, dst_iw_ipp);
    }

    auto start_iw_ipp = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++)
    {
        ipp::iwiAdd(img_iw_ipp, img_iw_ipp, dst_iw_ipp);
    }
    auto end_iw_ipp = std::chrono::high_resolution_clock::now();

    std::cout << "IW IPP Runtime : "
        << std::chrono::duration_cast<std::chrono::milliseconds>((end_iw_ipp - start_iw_ipp) / 100).count()
        << " ms"
        << std::endl;

    compare_result((Ipp8u*)dst_iw_ipp.m_ptr, dst_opencv);

    /* Add funciton implement with IPP API */
    ipp::Ipp8u* pSrc = img.data;
    int step8 = img.cols * sizeof(Ipp8u);
    ipp::Ipp8u* pDst = new Ipp8u[img.rows * img.cols * img.channels()];
    IppiSize roiSize = { img.cols, img.rows };
    int scaleFactor = 0;

    for (int i = 0; i < 100; i++) /* warm up */
    {
        ippiAdd_8u_C1RSfs(pSrc, step8, pSrc, step8, pDst, step8, roiSize, scaleFactor);
    }

    auto start_ipp = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++)
    {
        ippiAdd_8u_C1RSfs(pSrc, step8, pSrc, step8, pDst, step8, roiSize, scaleFactor);
    }
    auto end_ipp = std::chrono::high_resolution_clock::now();

    std::cout << "IPP Runtime : "
        << std::chrono::duration_cast<std::chrono::milliseconds>((end_ipp - start_ipp) / 100).count()
        << " ms"
        << std::endl;

    compare_result(pDst, dst_opencv); /* compare result of IPP with OpenCV one */

    return 0;
}