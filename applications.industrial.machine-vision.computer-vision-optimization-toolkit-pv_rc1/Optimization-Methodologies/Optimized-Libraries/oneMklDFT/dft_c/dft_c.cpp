//==============================================================
/* Copyright(C) 2022 Intel Corporation
* Licensed under the Intel Proprietary License
*/
// =============================================================


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "mkl_service.h"
#include "mkl_dfti.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using millis = std::chrono::milliseconds;
using namespace std;
using namespace cv;

static const bool DEBUG = false;
static const bool INFO = true;

int main(void)
{
    /* Pointer to input/output data */
    MKL_Complex8* data = NULL;

    /* Execution status */
    MKL_LONG status = 0;

    DFTI_DESCRIPTOR_HANDLE hand = NULL;

    char version[DFTI_VERSION_LENGTH];

    Mat img_in = imread("../data/input.jpg", IMREAD_GRAYSCALE);
    if (img_in.empty()) {
        printf(" read image failed \n");
    }


    img_in.convertTo(img_in, CV_32F, 1.0 / 255, 0);

    const int  width = img_in.cols;
    const int  height = img_in.rows;


    MKL_LONG N[2] = { height, width };

    Mat padded;
    int h = getOptimalDFTSize(img_in.rows);
    int w = getOptimalDFTSize(img_in.cols);

    copyMakeBorder(img_in, padded, 0, h - img_in.rows, 0, w - img_in.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(),CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);

    {
        auto start = std::chrono::high_resolution_clock::now();

        dft(complexI, complexI);
        idft(complexI, complexI);

        auto stop = std::chrono::high_resolution_clock::now();
        auto opencv_time = std::chrono::duration_cast<millis>(stop - start).count();

        if (INFO)
            printf("opencv   dft/idft took %ld ms \n", opencv_time);
    }

    DftiGetValue(0, DFTI_VERSION, version);

    status = DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_COMPLEX, 2, N);
    if (status != DFTI_NO_ERROR)
    {
        printf("error \n");
    }

    status = DftiCommitDescriptor(hand);
    if (status != DFTI_NO_ERROR) {
        printf("errot status=%d \n", status);
    }

    data = (MKL_Complex8*)mkl_malloc(w * h * sizeof(MKL_Complex8), 64);
    if (data == NULL) goto failed;

    for (int n2 = 0; n2 < h; n2++) {
        float* imgPtr = img_in.ptr<float>(n2);
        for (int n1 = 0; n1 < w; n1++) {
            int index = n2 * w + n1;
            data[index].real = (float)imgPtr[n1];
            data[index].imag = 0.0f;
        }
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        status = DftiComputeForward(hand, data);
        if (status != DFTI_NO_ERROR) goto failed;
        status = DftiComputeBackward(hand, data);
        if (status != DFTI_NO_ERROR) goto failed;

        auto stop = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<millis>(stop - start).count();
        if (INFO)
            printf("oneMKL C dft/idft took %ld ms\n", time);
    }

cleanup:
    DftiFreeDescriptor(&hand);
    mkl_free(data);
    return status;

failed:
    status = 1;
    goto cleanup;
}