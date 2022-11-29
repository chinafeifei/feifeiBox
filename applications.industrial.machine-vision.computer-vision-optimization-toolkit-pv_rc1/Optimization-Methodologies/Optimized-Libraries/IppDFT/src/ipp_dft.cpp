//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include <iostream>
#include <opencv2\opencv.hpp>
#include <ipps.h>
#include <ippi.h>
#include <ippcore.h>
#include <omp.h>
#include <Windows.h>

using namespace std;
using namespace cv;

#define HAVE_IPP 1

#define NTHREADS 2
#if 1
IppiDFTSpec_R_32f *pMemSpec;
void *pMemInit;
void *pMemBuffer[NTHREADS];
int size_Buffer;
static void dft_init(IppiSize &roiSize)
{
	int sizeSpec, sizeInit, sizeBuffer;
	/// get sizes for required buffers
	ippiDFTGetSize_R_32f(roiSize, IPP_FFT_DIV_INV_BY_N, ippAlgHintFast, &sizeSpec, &sizeInit,
						 &sizeBuffer);
	/// allocate memory for required buffers
	pMemSpec = (IppiDFTSpec_R_32f *)ippMalloc(sizeSpec);
	if (sizeInit > 0)
	{
		pMemInit = ippMalloc(sizeInit);
	}
	size_Buffer = sizeBuffer;
	if (sizeBuffer > 0)
	{
		for (int nth = 0; nth < NTHREADS; nth++)
		{
			pMemBuffer[nth] = ippMalloc(sizeBuffer);
		}
	}
	/// initialize DFT specification structure
	ippiDFTInit_R_32f(roiSize, IPP_FFT_DIV_INV_BY_N, ippAlgHintFast, pMemSpec, (Ipp8u *)pMemInit);
	if (sizeInit > 0)
	{
		ippFree(pMemInit);
	}
}

static void dft_cal_F(Ipp32f *pSrc, int srcStep, Ipp32f *pDst, int dstStep)
{
	/// perform forward DFT to put source data to frequency domain
	ippiDFTFwd_RToPack_32f_C1R(pSrc, srcStep, pDst, dstStep, pMemSpec, (Ipp8u *)(pMemBuffer[omp_get_thread_num() % NTHREADS]));
	/// ...
}

static void dft_cal(Ipp32f *pSrc, int srcStep, Ipp32f *pDst, int dstStep)
{
	/// perform forward DFT to put source data to frequency domain
	ippiDFTInv_PackToR_32f_C1R(pSrc, srcStep, pDst, dstStep, pMemSpec, (Ipp8u *)(pMemBuffer[omp_get_thread_num() % NTHREADS]));
	/// ...
}

static void dft_free()
{
	/// free buffers
	if (size_Buffer > 0)
	{
		for (int nth = 0; nth < NTHREADS; nth++)
		{
			ippFree(pMemBuffer[nth]);
		}
	}
	ippFree(pMemSpec);
}
#endif

static void ipp_dft(const Mat &in, const Size &dftSize, Mat &out)
{
	CV_Assert(in.type() == CV_8UC1);
	Mat src = in;
	Mat dst(out, Rect(0, 0, dftSize.width, dftSize.height));
	Mat dst1(out, Rect(0, 0, in.cols, in.rows));
	if (dst1.data != src.data)
		src.convertTo(dst1, dst1.depth());
	dft_cal_F((Ipp32f *)dst.data, dst.step, (Ipp32f *)dst.data, dst.step);
}

static void opencv_dft(const Mat &in, const Size &dftSize, Mat &out)
{
	CV_Assert(in.type() == CV_8UC1);
	Mat src = in;
	Mat dst(out, Rect(0, 0, dftSize.width, dftSize.height));
	Mat dst1(out, Rect(0, 0, in.cols, in.rows));
	if (dst1.data != src.data)
		src.convertTo(dst1, dst1.depth());
	dft(dst, dst, 0, out.rows);
}

int main()
{
	Mat src = imread("./src.bmp", 0);
	Size dftSize1(768, 512);
	IppiSize roi_Size;
	roi_Size.height = src.rows;
	roi_Size.width = src.cols;
	dft_init(roi_Size);
	int loop_num = 100;
	Mat output = Mat::zeros(src.size(), CV_32FC1);

	double t1 = getTickCount();
	for (int i = 0; i < loop_num; i++)
	{
#ifdef HAVE_IPP
		ipp_dft(src, src.size(), output);
#else
		opencv_dft(src, src.size(), output);
#endif
	}
	double t2 = getTickCount();
	double timeTotal = (t2 - t1) / getTickFrequency() * 1000;
	cout << "total time: " << timeTotal << endl;
	imshow("Image show", output);
	waitKey(0);
	system("pause");
	return 0;
}
