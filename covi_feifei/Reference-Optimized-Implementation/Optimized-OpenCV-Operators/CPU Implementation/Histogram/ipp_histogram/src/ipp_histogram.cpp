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
#include "base_ipp.h"

#include "ippcore.h"
#include "ipps.h"
#include "ippi.h"

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <sys/time.h>
#include <omp.h>

using namespace cv;
using namespace std;
using namespace std::chrono;

#define LOWER 0
#define UPPER 256
#define LEVELS 4
#define MAX_NUM_THREADS 32

static void printVersion()
{
    const IppLibraryVersion *pVersion;
    printf("\nIntel(R) IPP:\n");
    PRINT_LIB_VERSION(  , pVersion)
    PRINT_LIB_VERSION(s,  pVersion)
    PRINT_LIB_VERSION(i,  pVersion)
}

int histogram_openCV(Ipp8u* data, int width, int height)
{
    Mat src(height, width, CV_8UC1, data);
    int histSize = LEVELS; //level
    float range[] = { LOWER, UPPER }; //data range, the upper boundary is exclusive
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    Mat hist;
    auto start = system_clock::now();
    calcHist( &src, 1, 0, Mat(), hist, 1, &histSize, histRange, uniform, accumulate );
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    printf("histogram openCV takes %.2ld us ======\n", duration.count());

    printf("histogram_openCV: %d levels,\n", histSize);
    std::cout << hist;
    printf("\n");

    return EXIT_SUCCESS;
}

void histogram_ipp(Ipp8u* data, int width, int height)
{

    IppiSize roi = {width, height};
    IppStatus sts;

    const int nBins = LEVELS;
    int nLevels[] = { nBins+1 };
    Ipp32f lowerLevel[] = {LOWER};
    Ipp32f upperLevel[] = {UPPER};
    Ipp32f pLevels[nBins+1], *ppLevels[1];
    int sizeHistObj, sizeBuffer;

    IppiHistogramSpec* pHistObj;
    Ipp8u* pBuffer;
    Ipp32u pHistVec[nBins];
    int i;

    // get sizes for spec and buffer
    ippiHistogramGetBufferSize(ipp8u, roi, nLevels, 1/*nChan*/, 1/*uniform*/, &sizeHistObj, &sizeBuffer);

    pHistObj = (IppiHistogramSpec*)ippsMalloc_8u( sizeHistObj );
    pBuffer = (Ipp8u*)ippsMalloc_8u( sizeBuffer );
    // initialize spec
    ippiHistogramUniformInit( ipp8u, lowerLevel, upperLevel, nLevels, 1, pHistObj );

    // check levels of bins
    ppLevels[0] = pLevels;
    sts = ippiHistogramGetLevels( pHistObj, ppLevels );
    printf("\nhistogram_ipp, %d levels,\n", nBins);
    for (i = 0; i < nBins+1; i++)
        printf( "%.2f\t", pLevels[i]);

    auto start = system_clock::now();
    // calculate histogram
    sts = ippiHistogram_8u_C1R( data, width, roi, pHistVec, pHistObj, pBuffer );
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    printf("\nhistogram ipp takes %.2ld us\n", duration.count());

    ippsFree( pHistObj );
    ippsFree( pBuffer );

    printf("\nhist result:\n");
    for (i = 0; i < nBins; i++)
        printf( "%d\t", pHistVec[i]);

    printf("\n");
}

void histogramtile_ipp(Ipp8u* data, int width, int height)
{

    IppiSize roi1, roi2;
    
    IppiSize roi_temp = {width, height/MAX_NUM_THREADS};
    IppStatus sts;

    const int nBins = LEVELS;
    int nLevels[] = { nBins+1 };
    Ipp32f lowerLevel[] = {LOWER};
    Ipp32f upperLevel[] = {UPPER};
    Ipp32f pLevels[nBins+1], *ppLevels[1];
    int sizeHistObj, sizeBuffer;

    IppiHistogramSpec* pHistObj;
    Ipp8u* pBuffer;
    Ipp32u pHistVec[MAX_NUM_THREADS][nBins];
    Ipp32u pHistVecFinal[nBins] = {0};
    int i, k;

    int numThreads, slice, tail;
    int bufSize1, bufSize2;
    IppiSize dstTileSize, dstLastTileSize; 
    IppStatus pStatus[MAX_NUM_THREADS];

    // get sizes for spec and buffer
    ippiHistogramGetBufferSize(ipp8u, roi_temp, nLevels, 1/*nChan*/, 1/*uniform*/, &sizeHistObj, &sizeBuffer);
    pHistObj = (IppiHistogramSpec*)ippsMalloc_8u( sizeHistObj );
    // initialize spec
    ippiHistogramUniformInit( ipp8u, lowerLevel, upperLevel, nLevels, 1, pHistObj );

    // check levels of bins
    ppLevels[0] = pLevels;
    sts = ippiHistogramGetLevels( pHistObj, ppLevels );
    printf("\nhistogram_ipp, %d levels,\n", nBins);
    for (i = 0; i < nBins+1; i++)
        printf( "%.2f\t", pLevels[i]);
    printf("\n");

    auto start = system_clock::now();

    /* General transform function */
    /* Parallelized only by Y-direction here */
    #pragma omp parallel num_threads(MAX_NUM_THREADS)
    {
    #pragma omp master
        {
            numThreads = omp_get_num_threads();
            printf("omp thread %d\n", numThreads);

            slice = height / numThreads;
            tail  = (height % numThreads)? (height % numThreads):(height / numThreads);
            roi1 = {width, slice};
            roi2 = {width, tail};
            ippiHistogramGetBufferSize(ipp8u, roi1, nLevels, 1/*nChan*/, 1/*uniform*/, &sizeHistObj, &bufSize1);
            ippiHistogramGetBufferSize(ipp8u, roi2, nLevels, 1/*nChan*/, 1/*uniform*/, &sizeHistObj, &bufSize2);

            pBuffer = ippsMalloc_8u(bufSize1 * (numThreads - 1) + bufSize2);
        }

        #pragma omp barrier
        {
            if (pBuffer)
            {
                Ipp32u  j;
                Ipp8u  *pOneBuf;

                j = omp_get_thread_num();
                IppiSize roi = roi1;
                Ipp8u* pStart = data+j*width*slice;
                pOneBuf = pBuffer + j * bufSize1;

                if (j == numThreads - 1) 
                    roi = roi2;

                pStatus[j] = ippiHistogram_8u_C1R(pStart, width, roi, pHistVec[j], pHistObj, pOneBuf);
            }
        }
    }

    for (i = 0; i < nBins; i++)
        for (k = 0; k < numThreads; k++) {
            pHistVecFinal[i] += pHistVec[k][i];
    }

    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    printf("histogramtile ipp takes %.2ld us ======\n", duration.count());

    printf("hist ipp result:\n");
    for (i = 0; i < nBins; i++)
        printf( "%d\t", pHistVecFinal[i]);
    
    printf("\n\n");
}


int main(int argc, char *argv[])
{
    IppStatus status = ippStsNoErr;
    unsigned int resolution[2]  = {0, 0};
    unsigned int i;
    IppStatus sts;
    Ipp8u* data;
    unsigned int width = 4000;
    unsigned int height = 4000;

    // Cmd parsing
    const cmd::OptDef cmdOpts[] = {
        { 's', "", 2, cmd::KT_INTEGER, 0, &resolution[0], "fake image's width and height"},
        {0}
    };

    if(cmd::OptParse(argc, argv, cmdOpts))
    {
        PRINT_MESSAGE("invalid input parameters");
        return 1;
    }

    if (resolution[0] != 0 && resolution[1] != 0) {
        width = resolution[0];
        height = resolution[1];
    }

    // fill image with random values in [0..255] range with uniform distribution.
    IppsRandUniState_8u* pRndObj;
    int sizeRndObj;
    data = ippsMalloc_8u(width * height);

    // get spec size
    ippsRandUniformGetSize_8u( &sizeRndObj );
    pRndObj = (IppsRandUniState_8u*)ippsMalloc_8u( sizeRndObj );
    // initialize rnd spec
    ippsRandUniformInit_8u(pRndObj, 0/*low*/, 255/*high*/, 0/*seed*/ );

    // fill image
    for ( i=0; i<height; i++ ) {
        sts = ippsRandUniform_8u(data + i*width, width, pRndObj);
    }
    ippsFree( pRndObj );

    printVersion();
    printf("generated image of res %d * %d\n", width, height);

    histogramtile_ipp(data, width, height);
    histogram_openCV(data, width, height);
    return status;
}

