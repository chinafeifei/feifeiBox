//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include <opencv2/opencv.hpp>
#include <math.h>
#include <memory>
#include "ippcore.h"
#include "ipps.h"
#include "ippi.h"
#include "ippcv.h"
#include <omp.h>
#include <tbb/tbb.h>
#include <chrono>

using namespace tbb;
using timeunit = std::chrono::microseconds;

#define WARMUP_ROUNDS 20
#define COUNTING_ROUNDS 100

static const bool DEBUG = true;
static const bool SHOW_RESULT = false;
static const Ipp8u g_mask[] =
	{
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1};
static const IppiSizeL g_maskSize = {5, 5};

void compare(Ipp8u pSrc[], cv::Mat &smoothed)
{
	Ipp8u *pDst = smoothed.data;
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

	// printf("pixels same is %d, diff is %d \n", same, diff);
	if (DEBUG)
		printf("pixels correctness is %2.2f %% \n", correctness);
}

float calTime_opencv(cv::Mat &img, cv::Mat &out)
{

	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	for (auto i = 0; i < WARMUP_ROUNDS; i++)
	{
		// Execute MorphOpen
		cv::morphologyEx(img, out, cv::MORPH_OPEN, kernel);
	}

	/****Performance evaluation***/
	// Calculate time start,
	auto cv_start = std::chrono::high_resolution_clock::now();
	for (auto i = 0; i < COUNTING_ROUNDS; i++)
	{
		// Execute MorphOpen
		cv::morphologyEx(img, out, cv::MORPH_OPEN, kernel);
	}
	// Calculate time end
	auto cv_stop = std::chrono::high_resolution_clock::now();
	auto cv_time = std::chrono::duration_cast<timeunit>(cv_stop - cv_start).count() / 1000.0 / COUNTING_ROUNDS;

	std::cout << "CV Time consuming:" << cv_time << std::endl;

	if (SHOW_RESULT)
	{
		cv::namedWindow("Input image", 0);
		cv::imshow("Input image", img);
		cv::namedWindow("Processed image", 0);
		cv::imshow("Processed image", out);
		cv::waitKey(0);
	}

	return cv_time;
}
float calTime_ipp(cv::Mat &img, cv::Mat &out)
{

	/* Init IPP library */
	IppStatus status;
	ippInit();
	// Get source image and define dst image
	Ipp8u *pSrc = img.data;
	Ipp8u *ipp_dst = new Ipp8u[img.rows * img.cols * img.channels()]; // out.data;

	// Define ROI region
	// std::cout << img.cols << img.rows<<std::endl;
	IppiSizeL roiSize = {img.cols, img.rows};

	// Define srcStep (Distance, in bytes, between the starting points of consecutive lines in the source image.)
	int step8 = img.cols * sizeof(Ipp8u);

	// Define border type (ippBorderDefault,ippBorderRepl,ippBorderConst etc.)
	/*IppiBorderType borderType = ippBorderDefault;*/

	// Define pMorphSpec
	// AutoBuffer<IppiMorphAdvStateL> m_specOpen;
	IppiMorphAdvStateL *m_specOpen;
	IppSizeL specSize = 0;
	ippiMorphGetSpecSize_L(roiSize, g_maskSize, ipp8u, 1, &specSize);

	// m_specOpen.Alloc(1, specSize);
	m_specOpen = (IppiMorphAdvStateL *)malloc((int)(specSize * sizeof(IppiMorphAdvStateL *)));

	// initializes the internal state or specification structure
	status = ippiMorphInit_L(roiSize, g_mask, g_maskSize, ipp8u, 1, m_specOpen);

	// Generate buffer
	// AutoBuffer<Ipp8u> tmpBuffer;
	Ipp8u *tmpBuffer;
	IppSizeL tmpBufferSize = 0;
	ippiMorphGetBufferSize_L(roiSize, g_maskSize, ipp8u, 1, /* numChannels */ &tmpBufferSize);
	// tmpBuffer.Alloc(tmpBufferSize);
	// std::cout << "tmpBufferSize:" << tmpBufferSize<<std::endl;
	tmpBuffer = ippsMalloc_8u((int)tmpBufferSize);

	for (auto i = 0; i < WARMUP_ROUNDS; i++)
	{
		// Execute MorphOpen
		status = ippiMorphOpen_8u_C1R_L(pSrc, step8, ipp_dst, step8, roiSize, ippBorderDefault, NULL, m_specOpen, tmpBuffer);
	}
	/****Performance evaluation****/
	// Calculate time start, Performance evaluation
	auto ipp_start = std::chrono::high_resolution_clock::now();
	for (auto i = 0; i < COUNTING_ROUNDS; i++)
	{
		// Execute MorphOpen

		{
			status = ippiMorphOpen_8u_C1R_L(pSrc, step8, ipp_dst, step8, roiSize, ippBorderDefault, NULL, m_specOpen, tmpBuffer);
		}
	}
	// Calculate time end
	auto ipp_stop = std::chrono::high_resolution_clock::now();
	auto ipp_time = std::chrono::duration_cast<timeunit>(ipp_stop - ipp_start).count() / 1000.0 / COUNTING_ROUNDS;

	std::cout << "IPP Time consuming:" << ipp_time << std::endl;

	if (DEBUG)
		compare(ipp_dst, out);
	if (SHOW_RESULT)
	{
		cv::Mat show_img = cv::Mat(img.rows, img.cols, CV_8UC1, ipp_dst);
		cv::namedWindow("Input image", 0);
		cv::imshow("Input image", img);
		cv::namedWindow("Processed image", 0);
		cv::imshow("Processed image", show_img);
		cv::waitKey(0);
		// cv::imwrite("ippprocessed.bmp", show_img);
	}

	delete[] ipp_dst;
	ippsFree(tmpBuffer);
	// ippsFree(m_specOpen);
	return ipp_time;
}
void tbb_ipp_morphopen(cv::Mat &img, cv::Mat &out)
{

	const int THREAD_NUM = 4; // tbb 线程 slot 数量，在 4 核 8 线程的 cpu 上，该值最多为8
							  // 如果设置为 8 ，而 parallel_for 中的 blocked_range 为 0 ，WORKLOAD_NUM = 4
							  // 那么将会有 4 个 tbb 线程在工作，另外 4 个 tbb 线程空闲

	const int WORKLOAD_NUM = 4;				   // 将计算负载分割为多少份
	int chunksize = img.rows / WORKLOAD_NUM;   // 每个线程计算区域的行数
	IppiSizeL roiSize = {img.cols, chunksize}; // 每一个线程计算的区域大小

	/* Init IPP library */
	IppStatus status;
	ippInit();
	// Get source image and define dst image
	Ipp8u *pSrc = img.data;
	Ipp8u *ipp_dst = new Ipp8u[img.rows * img.cols * img.channels()]; // out.data;

	// Define srcStep (Distance, in bytes, between the starting points of consecutive lines in the source image.)
	int step8 = img.cols * sizeof(Ipp8u);

	// Define pMorphSpec
	// AutoBuffer<IppiMorphAdvStateL> m_specOpen;
	IppiMorphAdvStateL *m_specOpen;
	IppSizeL specSize = 0;
	ippiMorphGetSpecSize_L(roiSize, g_maskSize, ipp8u, 1, &specSize);

	// m_specOpen.Alloc(1, specSize);
	m_specOpen = (IppiMorphAdvStateL *)malloc((int)(specSize * sizeof(IppiMorphAdvStateL *)));

	// initializes the internal state or specification structure
	status = ippiMorphInit_L(roiSize, g_mask, g_maskSize, ipp8u, 1, m_specOpen);

	// Generate buffer
	// AutoBuffer<Ipp8u> tmpBuffer;
	IppSizeL tmpBufferSize = 0;
	ippiMorphGetBufferSize_L(roiSize, g_maskSize, ipp8u, 1, /* numChannels */ &tmpBufferSize);

	Ipp8u *pBufferArray[WORKLOAD_NUM];
	for (int i = 0; i < WORKLOAD_NUM; i++)
	{
		pBufferArray[i] = ippsMalloc_8u((int)tmpBufferSize);
	}

	tbb::task_arena no_hyper_thread_arena(tbb::task_arena::constraints{}.set_max_threads_per_core(1).set_max_concurrency(THREAD_NUM));

	// 限制每个物理核只能运行一个 tbb 线程，避免超线程影响
	// 设置 tbb 线程 slot 数量

	auto alltime = 0.0;
	const int LOOP_NUM = COUNTING_ROUNDS;
	for (int i = 0; i < LOOP_NUM; i++)
	{
		auto ipp_start = std::chrono::high_resolution_clock::now();

		no_hyper_thread_arena.execute([&]
									  { parallel_for(blocked_range<size_t>(0, WORKLOAD_NUM, 1),
													 [&](const blocked_range<size_t> &r)
													 {
														 // 这里一共有 WORKLOAD_NUM 个循环，由 tbb 自动分配给相应的线程执行
														 for (size_t i = r.begin(); i != r.end(); ++i)
														 {

															 Ipp8u *pSrcT; // 分别指向原矩阵的不同起始地址
															 Ipp8u *pDstT; // 分别指向目标矩阵的不同起始地址
															 IppStatus tStatus;
															 pSrcT = pSrc + step8 * chunksize * i;
															 pDstT = ipp_dst + step8 * chunksize * i;
															 // tBorder = ippBorderFirstStageInMem | ippBorderDefault | ippBorderInMemTop;
															 if (i == 0)
															 {
																 tStatus = ippiMorphOpen_8u_C1R_L(pSrcT, step8, pDstT, step8, roiSize, IppiBorderType(ippBorderFirstStageInMemBottom | ippBorderDefault), NULL, m_specOpen, pBufferArray[i]);
															 }
															 else if (i == 3)
															 {
																 tStatus = ippiMorphOpen_8u_C1R_L(pSrcT, step8, pDstT, step8, roiSize, IppiBorderType(ippBorderFirstStageInMemTop | ippBorderDefault), NULL, m_specOpen, pBufferArray[i]);
															 }
															 else
																 tStatus = ippiMorphOpen_8u_C1R_L(pSrcT, step8, pDstT, step8, roiSize, IppiBorderType(ippBorderFirstStageInMem | ippBorderDefault | ippBorderInMemTop | ippBorderInMemBottom), NULL, m_specOpen, pBufferArray[i]);
														 }
													 }); });

		auto ipp_stop = std::chrono::high_resolution_clock::now();
		auto ipp_time = std::chrono::duration_cast<timeunit>(ipp_stop - ipp_start).count() / 1000.0;

		if (i < WARMUP_ROUNDS)
			continue;
		alltime += ipp_time;
	}

	std::cout << "tbb ipp time consuming:" << alltime / (LOOP_NUM - WARMUP_ROUNDS) << std::endl;

	if (DEBUG)
		compare(ipp_dst, out);

	delete ipp_dst;
	for (int i = 0; i < WORKLOAD_NUM; i++)
	{
		ippsFree(pBufferArray[i]);
	}
}

int main()
{

	// READ image in Grayscale
	cv::Mat img = cv::imread("data/4.bmp", 0);
	cv::Mat reImg;
	cv::Mat out;
	float time_ipp;
	float time_cv;
	/*cv::resize(img, reImg, cv::Size(), 2, 4);*/

#ifdef _WIN32
	HANDLE mHandle = GetCurrentProcess();
	BOOL result = SetPriorityClass(mHandle, REALTIME_PRIORITY_CLASS);
	SetThreadAffinityMask(GetCurrentThread(), 0x00000001); // 0x01，Core0
#endif

	std::cout << "/************************Performance evaluation starts************************/" << std::endl;
	time_cv = calTime_opencv(img, out);
	time_ipp = calTime_ipp(img, out);
	tbb_ipp_morphopen(img, out);
	std::cout << "/************************Performance evaluation ends************************/" << std::endl;

	return 0;
}
