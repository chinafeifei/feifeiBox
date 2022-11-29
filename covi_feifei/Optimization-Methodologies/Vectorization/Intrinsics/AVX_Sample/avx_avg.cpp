//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include <iostream>
#include <time.h>
#include <cstdlib>
#include <ctime>

#include <immintrin.h>

#define _countof(array) (sizeof(array) / sizeof(array[0]))

static const int length = 1024 * 8;
static float a[length];
static int preventOptimize = 0;
static const int compute_times = 100000;

/*original implementation*/
float scalarAverage()
{
	preventOptimize++;
	float sum = 0.0;
	for (uint32_t j = 0; j < _countof(a); ++j)
	{
		sum += a[j];
	}
	return sum / _countof(a);
}

/*AVX2 implementation*/
float avxAverage()
{
	preventOptimize++;
	__m256 sumx8 = _mm256_setzero_ps();
	for (uint32_t j = 0; j < _countof(a); j = j + 8)
	{
		sumx8 = _mm256_add_ps(sumx8, _mm256_loadu_ps(&(a[j])));
	}

#ifdef __linux__
	float sum = sumx8[0] + sumx8[1] +
				sumx8[2] + sumx8[3] +
				sumx8[4] + sumx8[5] +
				sumx8[6] + sumx8[7];
#else
	float sum = sumx8.m256_f32[0] + sumx8.m256_f32[1] +
				sumx8.m256_f32[2] + sumx8.m256_f32[3] +
				sumx8.m256_f32[4] + sumx8.m256_f32[5] +
				sumx8.m256_f32[6] + sumx8.m256_f32[7];
#endif
	return sum / _countof(a);
}

/*AVX512 implementation*/
float avx512AverageKernel()
{
	preventOptimize++;
	__m512 sumx16 = _mm512_setzero_ps();
	for (uint32_t j = 0; j < _countof(a); j = j + 16)
	{
		sumx16 = _mm512_add_ps(sumx16, _mm512_loadu_ps(&(a[j])));
	}
	float sum = _mm512_reduce_add_ps(sumx16);
	return sum / _countof(a);
}

int main()
{
	for (int i = 0; i < length; i++)
	{
		a[i] = 10 + (float)(rand()) / RAND_MAX * (10000 - 10);
	}

	float avg = 0;
	clock_t t0 = clock();
	for (int cnt = 0; cnt < compute_times; cnt++)
	{
		avg += scalarAverage();
	}
	clock_t t1 = clock();
	std::cout << "Scalar average: " << avg << std::endl;
	std::cout << "==================Original time (ms): " << (float)(t1 - t0) * 1000 / CLOCKS_PER_SEC << std::endl;

	float avxavg = 0;
	clock_t t2 = clock();
	for (int cnt = 0; cnt < compute_times; cnt++)
	{
		avxavg += avxAverage();
	}
	clock_t t3 = clock();
	std::cout << "AVX2 average: " << avxavg << std::endl;
	std::cout << "==================AVX2 accelerated (ms): " << (float)(t3 - t2) * 1000 / CLOCKS_PER_SEC << std::endl;

	float avx512avg = 0;
	clock_t t4 = clock();
	for (int cnt = 0; cnt < compute_times; cnt++)
	{
		avx512avg += avx512AverageKernel();
	}
	clock_t t5 = clock();
	std::cout << "AVX512 average: " << avx512avg << std::endl;
	std::cout << "==================AVX512 accelerated (ms): " << (float)(t5 - t4) * 1000 / CLOCKS_PER_SEC << std::endl;

	return 0;
}