#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cudaWarp.h"

using namespace cv;

void warpAffineMigrated(cv::Mat &src, cv::Mat &dst, cv::Mat M);
void warpAffineMigratedMat(cv::Mat &srcMat, cv::Mat &dstMat, cv::Mat M);

int main(int argc, char **argv)
{
    if (argc > 2)
    {
        printf("Usage:");
        printf("     %s opencv\n", argv[0]);
        printf("     %s \n", argv[0]);
    }
    cv::Mat srcMat = imread("./data/color_4288.jpg");
    if (srcMat.empty())
    {
        printf("srcMat is empty\n");
        return -1;
    }
    cv::Mat dstMat = srcMat.clone();
    cv::Mat dstMat_dpcpp = srcMat.clone();

    const cv::Point2f src_pt[] = {
        cv::Point2f(200, 200),
        cv::Point2f(250, 200),
        cv::Point2f(200, 100)};

    const cv::Point2f dst_pt[] = {
#if 0
        cv::Point2f(200,200),
        cv::Point2f(250,200),
        cv::Point2f(200,100)
#else
        cv::Point2f(300, 100),
        cv::Point2f(300, 50),
        cv::Point2f(200, 100)
#endif
    };

    const cv::Mat affine_matrix = cv::getAffineTransform(src_pt, dst_pt);

    printf("Run warpAffineMigrated\n");
    warpAffineMigratedMat(srcMat, dstMat_dpcpp, affine_matrix);

    printf("Run cv::warpAffine\n");
    cv::warpAffine(srcMat, dstMat, affine_matrix, srcMat.size());

    cv::imwrite("./data/dstMat.ppm", dstMat);
    cv::imwrite("./data/dstMat_dpcpp.ppm", dstMat_dpcpp);

    // imshow("src", srcMat);
    // imshow("dst", dstMat);

    // waitKey(0);
    return 0;
}

void warpAffineMigrated(cv::Mat &src, cv::Mat &dst, cv::Mat M)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();

    // Property of srcMat
    int imWidth = src.cols;
    int imHeight = src.rows;
    size_t sizeInBytes = src.total() * src.elemSize();

    // Tranformation Matrix to 2D-array
    float transform_array[2][3];
    double *fMatrix = (double *)M.data;

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            transform_array[i][j] = *(fMatrix++);

    // Expand the size by filling zeros
    size_t sizeInInt = sizeInBytes / 3 * 4;
    unsigned char *tBuffer = new unsigned char[sizeInInt]();
    for (int i = 0, j = 0; i < sizeInBytes;)
    {
        memcpy(&tBuffer[j], &src.data[i], 3);
        i += 3;
        j += 4;
    }

    unsigned char *in_ctl1 = sycl::malloc_device<unsigned char>(sizeInInt, q_ct1);
    unsigned char *out_ctl1 = sycl::malloc_device<unsigned char>(sizeInInt, q_ct1);
    // Copy data from host to GPU mem
    q_ct1.memcpy(in_ctl1, tBuffer, sizeInInt).wait();

    // Run cudaWarpAffine
    cudaWarpAffine((sycl::uchar4 *)in_ctl1, (sycl::uchar4 *)out_ctl1, imWidth,
                   imHeight, transform_array, false);

    // Copy back
    q_ct1.memcpy(tBuffer, out_ctl1, sizeInInt).wait();
    for (int i = 0, j = 0; i < sizeInBytes;)
    {
        memcpy(&dst.data[i], &tBuffer[j], 3);
        i += 3;
        j += 4;
    }
}

void warpAffineMigratedMat(cv::Mat &srcMat, cv::Mat &dstMat, cv::Mat M)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();

    cv::Mat src;
    cv::cvtColor(srcMat, src, COLOR_BGR2BGRA);
    cv::Mat dst = src.clone();
    // Property of srcMat
    int imWidth = src.cols;
    int imHeight = src.rows;
    size_t sizeInBytes = src.total() * src.elemSize();

    // Tranformation Matrix to 2D-array
    float transform_array[2][3];
    double *fMatrix = (double *)M.data;

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            transform_array[i][j] = *(fMatrix++);

    unsigned char *in_ctl1 = sycl::malloc_device<unsigned char>(sizeInBytes, q_ct1);
    unsigned char *out_ctl1 = sycl::malloc_device<unsigned char>(sizeInBytes, q_ct1);
    // Copy data from host to GPU mem
    q_ct1.memcpy(in_ctl1, src.data, sizeInBytes).wait();

    // Run cudaWarpAffine
    cudaWarpAffine((sycl::uchar4 *)in_ctl1, (sycl::uchar4 *)out_ctl1, imWidth,
                   imHeight, transform_array, false);

    // Copy back
    q_ct1.memcpy(dst.data, out_ctl1, sizeInBytes).wait();
    cv::cvtColor(dst, dstMat, COLOR_BGRA2BGR);
}
