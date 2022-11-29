/*******************************************************************************
* Copyright 2012-2021 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/
#include "base.h"
#include "base_image.h"

#include "ippcore.h"
#include "ipps.h"
#include "ippi.h"

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

#include <time.h>
#include <omp.h>

using namespace cv;
using namespace std;
using namespace std::chrono;
//#define SINGLE_THREAD


const char* source_window = "Source image";
const char* warp_window = "Warp";
double coeffs[2][3] =  {{1.0, 0.5, 0.0}, {0.5, 1.0, 0.0}};
#define FILE_NAME "blobX4.png"
#define IMG_COLOR 100
#define MAX_NUM_THREADS 16

IppStatus warpAffine_openCV(unsigned int* res)
{
  Mat warp_mat( 2, 3, CV_64FC1, coeffs);
  Mat src, warp_dst;
  Ipp8u* data;
  int numchanels = 1;

  if (res[0] != 0 && res[1] != 0) {
      data = ippsMalloc_8u(res[0] * res[1] * numchanels);
      memset(data, IMG_COLOR, res[0] * res[1] * numchanels); //creat a picture with some color, for example gray
      Mat srcfake(res[1], res[0], CV_8UC1, data);
      warp_dst = Mat::zeros( srcfake.rows, srcfake.cols, srcfake.type() );
      printf("resolution: %d * %d, openCV fake image\n", srcfake.cols, srcfake.rows);

      auto start = system_clock::now();
      warpAffine( srcfake, warp_dst, warp_mat, warp_dst.size() );

      auto end = system_clock::now();
      auto duration = duration_cast<microseconds>(end - start);
      printf("warpAffine openCV takes %.2ld us\n", duration.count());
  } else {
      src = imread(FILE_NAME, IMREAD_GRAYSCALE);
      warp_dst = Mat::zeros( src.rows, src.cols, src.type() );
      printf("resolution: %d * %d, channels: %d, openCV image\n", src.cols, src.rows, src.channels());

      auto start = system_clock::now();
      warpAffine( src, warp_dst, warp_mat, warp_dst.size() );

      auto end = system_clock::now();
      auto duration = duration_cast<microseconds>(end - start);
      printf("warpAffine openCV takes %.2ld us\n", duration.count());
  }
  imwrite("warp_opencv.png", warp_dst);

  return 0;
}

IppStatus warpAffine_ipp(Ipp8u* pSrc, IppiSize srcSize, Ipp32s srcStep, Ipp8u* pDst, IppiSize dstSize,
    Ipp32s dstStep, double coeffs[2][3])
{
    IppiWarpSpec* pSpec = 0;
    int specSize = 0, initSize = 0, bufSize = 0; Ipp8u* pBuffer  = 0;
    const Ipp32u numChannels = 1;
    IppiPoint dstOffset = {0, 0};
    IppStatus status = ippStsNoErr;
    IppiBorderType borderType = ippBorderConst;
    IppiWarpDirection direction = ippWarpForward;
    Ipp64f pBorderValue[numChannels];

    for (int i = 0; i < numChannels; ++i) pBorderValue[i] = 255.0;

    auto start = system_clock::now();

    /* Spec and init buffer sizes */
    status = ippiWarpAffineGetSize(srcSize, dstSize, ipp8u, coeffs, ippLinear, direction, borderType, &specSize, &initSize);

    if (status != ippStsNoErr) return status;

    /* Memory allocation */
    pSpec = (IppiWarpSpec*)ippsMalloc_8u(specSize);

    if (pSpec == NULL)
    {
        return ippStsNoMemErr;
    }

    /* Filter initialization */
    status = ippiWarpAffineLinearInit(srcSize, dstSize, ipp8u, coeffs, direction, numChannels, borderType, pBorderValue, 0, pSpec);

    if (status != ippStsNoErr)
    {
        ippsFree(pSpec);
        return status;
    }

    /* work buffer size */
    status = ippiWarpGetBufferSize(pSpec, dstSize, &bufSize);
    if (status != ippStsNoErr)
    {
        ippsFree(pSpec);
        return status;
    }

    pBuffer = ippsMalloc_8u(bufSize);
    if (pBuffer == NULL)
    {
        ippsFree(pSpec);
        return ippStsNoMemErr;
    }

    /* Resize processing */
    status = ippiWarpAffineLinear_8u_C1R(pSrc, srcStep, pDst, dstStep, dstOffset, dstSize, pSpec, pBuffer);

    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    printf("warpAffine ipp takes %.2ld us\n\n", duration.count());
    ippsFree(pSpec);
    ippsFree(pBuffer);

    return status;
}

IppStatus warpAffinetile_ipp(Ipp8u* pSrc, IppiSize srcSize, Ipp32s srcStep, Ipp8u* pDst, IppiSize dstSize, Ipp32s dstStep, const double coeffs[2][3])
{
    IppiWarpSpec* pSpec = 0;
    int specSize = 0, initSize = 0, bufSize = 0; Ipp8u* pBuffer  = 0;
    Ipp8u* pInitBuf = 0;
    const Ipp32u numChannels = 1;
    IppiPoint dstOffset = {0, 0};
    IppiPoint srcOffset = {0, 0};
    IppStatus status = ippStsNoErr;
    IppiBorderType borderType = ippBorderConst;
    IppiWarpDirection direction = ippWarpForward;
    int numThreads, slice, tail;
    int bufSize1, bufSize2;
    IppiSize dstTileSize, dstLastTileSize; IppStatus pStatus[MAX_NUM_THREADS];
    Ipp64f pBorderValue[numChannels];

    for (int i = 0; i < numChannels; ++i) pBorderValue[i] = 255.0;

    /* Spec and init buffer sizes */
    status = ippiWarpAffineGetSize(srcSize, dstSize, ipp8u, coeffs, ippLinear, direction, borderType, &specSize, &initSize);

    if (status != ippStsNoErr) return status;

    /* Memory allocation */
    pSpec = (IppiWarpSpec*)ippsMalloc_8u(specSize);

    if (pSpec == NULL)
    {
        return ippStsNoMemErr;
    }

    /* Filter initialization */
    status = ippiWarpAffineLinearInit(srcSize, dstSize, ipp8u, coeffs, direction, numChannels, borderType, pBorderValue, 0, pSpec);

    if (status != ippStsNoErr)
    {
        ippsFree(pSpec);
        return status;
    }
    auto start = system_clock::now();

    /* General transform function */
    /* Parallelized only by Y-direction here */
    #pragma omp parallel num_threads(MAX_NUM_THREADS)
    {
    #pragma omp master
        {
            numThreads = omp_get_num_threads();
            printf("=======omp thread %d\n", numThreads);

            slice = dstSize.height / numThreads; tail  = dstSize.height % numThreads;

            dstTileSize.width = dstLastTileSize.width = dstSize.width;
            dstTileSize.height = slice;
            dstLastTileSize.height = slice + tail;

            ippiWarpGetBufferSize(pSpec, dstTileSize, &bufSize1);
            ippiWarpGetBufferSize(pSpec, dstLastTileSize, &bufSize2);

            pBuffer = ippsMalloc_8u(bufSize1 * (numThreads - 1) + bufSize2);
        }

        #pragma omp barrier
        {
            if (pBuffer)
            {
                Ipp32u  i;
                Ipp8u  *pDstT; Ipp8u  *pOneBuf;
                IppiPoint srcOffset = {0, 0};
                IppiPoint dstOffset = {0, 0};
                IppiSize  srcSizeT = srcSize; IppiSize  dstSizeT = dstTileSize;

                i = omp_get_thread_num();

                dstSizeT.height = slice; dstOffset.y += i * slice;

                if (i == numThreads - 1) dstSizeT = dstLastTileSize;

                pDstT = (Ipp8u*)((char*)pDst + dstOffset.y * dstStep);

                pOneBuf = pBuffer + i * bufSize1;

                pStatus[i] = ippiWarpAffineLinear_8u_C1R (pSrc, srcStep, pDstT, dstStep, dstOffset, dstSizeT, pSpec, pOneBuf);
            }
        }
    }
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    printf("warpAffinetile ipp takes %.2ld us\n\n", duration.count());

    ippsFree(pSpec);

    if (pBuffer == NULL) return ippStsNoMemErr;

    ippsFree(pBuffer);

    for (Ipp32u i = 0; i < numThreads; ++i)
    {
        /* Return bad status */
        if(pStatus[i] != ippStsNoErr) return pStatus[i];
    }

    return status;
}

int main(int argc, char *argv[])
{
    IppStatus status = ippStsNoErr;
    unsigned int resolution[2]  = {0, 0};

    Ipp8u* pSrc; 
    IppiSize srcSize;
    Ipp32s srcStep;
    Ipp8u* pDst;
    IppiSize dstSize;
    Ipp32s dstStep;
    Ipp8u* data;
    int channels = 1;
    status = warpAffine_openCV(resolution);
    Sleep(1000);

        Mat img = imread(FILE_NAME, IMREAD_GRAYSCALE);
        printf("\nresolution: %d * %d, channels: %d, IPP image\n", img.cols, img.rows, img.channels());
        pSrc = img.data;
        srcSize = {img.cols, img.rows};
        dstSize = {img.cols, img.rows};
        srcStep = img.cols * img.channels();
        dstStep = img.cols * img.channels();
        pDst = ippsMalloc_8u(dstSize.width * dstSize.height * img.channels());
#ifdef SINGLE_THREAD
        status = warpAffine_ipp(pSrc, srcSize, srcStep, pDst, dstSize,dstStep, coeffs);
#else
        status = warpAffinetile_ipp(pSrc, srcSize, srcStep, pDst, dstSize, dstStep, coeffs);
#endif

    Mat output(dstSize.height, dstSize.width, CV_8UC1, pDst);
    //imshow("warpAffine_ipp", output);
    imwrite("warp_ipp.png", output);
    //waitKey(0);

    if(status != 0)
        printf("error happen, %d\n", status);

    return status;
}
