//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================

#include <opencv2/opencv.hpp>
#include <math.h>
#include <memory>
#include "ippcore_tl.h"
#include "ippi.h"
#include "ippi_tl.h"
#include <omp.h>
#include <tbb/tbb.h>

using namespace tbb;

void ipp_single_thread(cv::Mat &img)
{
	/*Example code for single thread ipp library implementation*/

	std::cout << "This an ipp single thread example" << std::endl;

	Ipp8u *src1 = img.data;
	Ipp8u *src2 = img.data;
	Ipp8u *ipp_dst = new Ipp8u[img.rows * img.cols * img.channels()];
	IppiSizeL roiSize = {img.cols, img.rows};
	int step8 = img.cols * sizeof(Ipp8u);
	int scaleFactor = 0; // later examples for 2 and -2 values

	ippiAdd_8u_C1RSfs_L(src1, step8, src2, step8, ipp_dst, step8, roiSize, scaleFactor);

	std::cout << "This an ipp single thread example : Done, Great !" << std::endl;
}

void ipp_multi_thread_TL(cv::Mat &img)
{
	/*Example code for multi thread ipp library implementation: Using TL layer*/

	std::cout << "This an ipp multi thread example: Using TL layer" << std::endl;
	Ipp8u *src1 = img.data;
	Ipp8u *src2 = img.data;
	Ipp8u *ipp_dst = new Ipp8u[img.rows * img.cols * img.channels()];
	IppiSizeL roiSize = {img.cols, img.rows};
	int step8 = img.cols * sizeof(Ipp8u);
	int scaleFactor = 0;

	int num[] = {0};
	ippGetNumThreads_LT(num);
	/*std::cout << "ipp multi thread: " << *num << std::endl;*/
	ippiAdd_8u_C1RSfs_LT(src1, step8, src2, step8, ipp_dst, step8, roiSize, scaleFactor);
	std::cout << "This an ipp multi thread example: Using TL layer, Done, Great !" << std::endl;
}

void ipp_multi_thread_omp(cv::Mat &img)
{
	/*Example code for multi thread ipp library implementation: Using OpenMP*/

	std::cout << "This an ipp multi thread example: Using OpenMP" << std::endl;
	Ipp8u *src1 = img.data;
	Ipp8u *src2 = img.data;
	Ipp8u *ipp_dst = new Ipp8u[img.rows * img.cols * img.channels()];
	IppiSizeL roiSize;
	int step8 = img.cols * sizeof(Ipp8u);
	int scaleFactor = 0;

	int chunksize;
	int numThreads = 8;
	chunksize = img.rows / numThreads;
	roiSize = {img.cols, chunksize};

#pragma omp parallel num_threads(8)
	{
#pragma omp master
		{
			int numThreads = omp_get_num_threads();
		}

#pragma omp barrier
		{

			{
				/*Split images to subimages and operate on subimages*/
				Ipp8u *pSrcT;
				Ipp8u *pDstT;
				IppStatus tStatus;
				int i = omp_get_thread_num();
				pSrcT = src1 + step8 * chunksize * i;
				pDstT = ipp_dst + step8 * chunksize * i;
				ippiAdd_8u_C1RSfs_L(pSrcT, step8, pSrcT, step8, pDstT, step8, roiSize, scaleFactor);
			}
		}
	}

	std::cout << "This an ipp multi thread example: Using OpenMP, Done, Great ! " << std::endl;
}

void ipp_multi_thread_tbb(cv::Mat &img)
{
	/*Example code for multi thread ipp library implementation: Using TBB*/

	std::cout << "This an ipp multi thread example: Using TBB" << std::endl;

	Ipp8u *src1 = img.data;
	Ipp8u *src2 = img.data;
	Ipp8u *ipp_dst = new Ipp8u[img.rows * img.cols * img.channels()];
	int step8 = img.cols * sizeof(Ipp8u);
	int scaleFactor = 0;

	const int THREAD_NUM = 4;

	const int WORKLOAD_NUM = 4;
	int chunksize = img.rows / WORKLOAD_NUM;
	IppiSizeL roiSize = {img.cols, chunksize}; // ROI for each thread

	tbb::task_arena no_hyper_thread_arena(tbb::task_arena::constraints{}.set_max_threads_per_core(1).set_max_concurrency(THREAD_NUM));

	no_hyper_thread_arena.execute([&]
								  { parallel_for(blocked_range<size_t>(0, WORKLOAD_NUM, 1),
												 [&](const blocked_range<size_t> &r)
												 {
													 // Loop for # of threads times
													 for (size_t i = r.begin(); i != r.end(); ++i)
													 {
														 /*Split images to subimages and operate on subimages*/
														 Ipp8u *pSrcT;
														 Ipp8u *pDstT;
														 IppStatus tStatus;
														 pSrcT = src1 + step8 * chunksize * i;
														 pDstT = ipp_dst + step8 * chunksize * i;
														 ippiAdd_8u_C1RSfs_L(pSrcT, step8, pSrcT, step8, pDstT, step8, roiSize, scaleFactor);
													 }
												 }); });

	std::cout << "This an ipp multi thread example: Using TBB, Done, Great !" << std::endl;
}
int main()
{

	// READ image in Grayscale
	cv::Mat img = cv::imread("data/4.bmp", 0);

	std::cout << "/************************Ipp Single & Multi thread example************************/" << std::endl;
	ipp_single_thread(img);
	ipp_multi_thread_TL(img);
	ipp_multi_thread_omp(img);
	ipp_multi_thread_tbb(img);
	std::cout << "/************************Ipp Single & Multi thread example************************/" << std::endl;

	return 0;
}
