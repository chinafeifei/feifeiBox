//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
 // =============================================================
#pragma once
#include <iostream>
#include <fstream>
#include "ippcore.h"
#include "ipps.h"
#include "ippi.h"
#include "ippcv.h"
#include "ippcc.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>

using millis = std::chrono::milliseconds;
using namespace std;
using namespace cv;

static const bool DEBUG = false;
static const bool INFO = true;

#define EXAMPLE_IMG_PATH "./data/color_4288.bmp"
// Simplified BMP structure.
// See http://msdn.microsoft.com/en-us/library/dd183392(v=vs.85).aspx
#pragma pack(push, 1)
struct bmp_header
{
    char bf_type[2];
    unsigned int bf_size;
    unsigned int bf_reserved;
    unsigned int bf_off_bits;

    unsigned int bi_size;
    unsigned int bi_width;
    unsigned int bi_height;
    unsigned short bi_planes;
    unsigned short bi_bit_count;
    unsigned int bi_compression;
    unsigned int bi_size_image;
    unsigned int bi_x_pels_per_meter;
    unsigned int bi_y_pels_per_meter;
    unsigned int bi_clr_used;
    unsigned int bi_clr_important;
};

struct Image
{
    unsigned int size;
    unsigned int width;
    unsigned int height;
    int bpc;
    char* data;
};

void bmp_write(std::string fname, int h, int w, int ldw, char* data)
{
    unsigned sizeof_line = (w * 3 + 3) / 4 * 4;
    unsigned sizeof_image = h * sizeof_line;

    bmp_header header = { {'B', 'M'},
                         unsigned(sizeof(header) + sizeof_image),
                         0,
                         sizeof(header),
                         sizeof(header) - offsetof(bmp_header, bi_size),
                         unsigned(w),
                         unsigned(h),
                         1,
                         24,
                         0,
                         sizeof_image,
                         6000,
                         6000,
                         0,
                         0 };

    std::fstream fp;
    fp.open(fname, std::fstream::out | std::fstream::binary);
    if (fp.fail())
        printf("failed to save the image, cannot open file", fname.c_str());

    fp.write((char*)(&header), sizeof(header));

    fp.write((char*)(data), sizeof_image);

    fp.close();
}

// Read image from fname and convert it to gray-scale REAL.
int bmp_read(Image* image, std::string fname)
{
    std::fstream fp;
    char* data;
    fp.open(fname, std::fstream::in | std::fstream::binary);
    if (fp.fail())
        return -1;

    bmp_header header;

    fp.read((char*)(&header), sizeof(header));
    if (header.bi_bit_count != 24)
        printf("not a 24-bit image in %s\n", fname.c_str());
    if (header.bi_compression)
        printf("%s is compressed bmp\n", fname.c_str());

    image->width = header.bi_width;
    image->height = header.bi_height;
    image->bpc = header.bi_bit_count / 8;
    image->size = header.bi_width * header.bi_height * header.bi_bit_count / 8;

    data = (char*)malloc(image->size);

    fp.seekg(sizeof(header), std::ios_base::beg);

    fp.read((char*)data, image->size);

    image->data = data;
    fp.close();
    return 0;
}


int OpenCV_Filp()
{
    Mat src = imread(EXAMPLE_IMG_PATH);

    if (src.empty())
    {
        printf("OpenCV read image failed! \n");
        return -1;
    }

    Mat FlipedImg;

    flip(src, FlipedImg, -1);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++)
        flip(src, FlipedImg, -1);
    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<millis>(stop - start).count();

    if (INFO)
        printf("OpenCV Image Flip took %ld ms \n", time);

    //namedWindow("Flip", WINDOW_AUTOSIZE);
    //imshow("Flip", FlipedImg);
    //waitKey(0);

    return 0;
}

typedef int Status;

Status IPP_Filp(void) {
    IppStatus ippSts = ippStsNoErr;
    Image SrcImage, DstImage;

    IppiSize roi;

    Image* pSrcImage = &SrcImage;
    Image* pDstImage = &DstImage;

    if (bmp_read(pSrcImage, EXAMPLE_IMG_PATH)) {
        printf("IPP open file failed \n");
        return -1;
    }

    roi.width = pSrcImage->width;
    roi.height = pSrcImage->height;
    pDstImage->width = pSrcImage->width;
    pDstImage->height = pSrcImage->height;
    pDstImage->bpc = pSrcImage->bpc;
    pDstImage->data = (char*)malloc(pSrcImage->size);
    if (!pSrcImage || !pDstImage)
        return ippStsNoMemErr;

    ippiMirror_8u_C3R((Ipp8u*)pSrcImage->data, (int)pSrcImage->width * pSrcImage->bpc, (Ipp8u*)pDstImage->data, pDstImage->width * pDstImage->bpc, roi, ippAxsHorizontal);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++)
        ippSts = ippiMirror_8u_C3R((Ipp8u*)pSrcImage->data, (int)pSrcImage->width * pSrcImage->bpc, (Ipp8u*)pDstImage->data, pDstImage->width * pDstImage->bpc, roi, ippAxsHorizontal);
    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<millis>(stop - start).count();

    if (INFO)
        printf("IPP    Image Flip took %ld ms\n", time);

   // bmp_write("Flip.bmp", pDstImage->height, pDstImage->width, pDstImage->width, pDstImage->data);

    free(pDstImage->data);
    free(pSrcImage->data);
    return 0;
}
int main()
{

    IPP_Filp();
    OpenCV_Filp();
    return 0;
}
