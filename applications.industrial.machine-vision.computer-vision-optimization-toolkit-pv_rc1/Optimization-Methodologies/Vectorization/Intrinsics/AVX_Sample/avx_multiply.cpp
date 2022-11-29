//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include <iostream>
#include <time.h>
#include <immintrin.h>

#include <cstdlib>
#include <ctime>

// scalar multiply

static const int row = 16;
static const int col = 4096;
static float v1[row * col];
static float v2[col];
static float out[row];
static const int compute_times = 1000;

/*original implementation*/
static float *scalarMultiply()
{
	for (uint64_t i = 0; i < row; i++)
	{
		float sum = 0;
		for (uint64_t j = 0; j < col; j++)
			sum = sum + v1[i * col + j] * v2[j];
		out[i] = sum;
	}

	return out;
}

/*AVX2 implementation*/
static float outx8[row];
static float *avxMultiply()
{
	for (uint64_t i = 0; i < row; i++)
	{
		__m256 mulx8 = _mm256_set1_ps(0.0);
		__m256 sumx8 = _mm256_set1_ps(0.0);
		for (uint64_t j = 0; j < col; j += 8)
		{
			__m256 a = _mm256_loadu_ps(&(v1[i * col + j]));
			__m256 b = _mm256_loadu_ps(&(v2[j]));
			mulx8 = _mm256_mul_ps(a, b);
			sumx8 = _mm256_add_ps(mulx8, sumx8);
		}
#ifdef __linux__
		outx8[i] = sumx8[0] + sumx8[1] +
				   sumx8[2] + sumx8[3] +
				   sumx8[4] + sumx8[5] +
				   sumx8[6] + sumx8[7];
#else
		outx8[i] = sumx8.m256_f32[0] + sumx8.m256_f32[1] +
				   sumx8.m256_f32[2] + sumx8.m256_f32[3] +
				   sumx8.m256_f32[4] + sumx8.m256_f32[5] +
				   sumx8.m256_f32[6] + sumx8.m256_f32[7];
#endif
	}

	return outx8;
}

/*AVX512 implementation*/
static float outx16[row];
static float *avx512Multiply()
{
	for (uint64_t i = 0; i < row; i++)
	{
		__m512 sumx16 = _mm512_set1_ps(0.0);
		for (uint64_t j = 0; j < col; j += 16)
		{
			__m512 a = _mm512_loadu_ps(&(v1[i * col + j]));
			__m512 b = _mm512_loadu_ps(&(v2[j]));
			sumx16 = _mm512_fmadd_ps(a, b, sumx16);
		}
		outx16[i] = _mm512_reduce_add_ps(sumx16);
	}
	return outx16;
}

int main()
{

	// scalar multiply
	std::cout << "Scalar multiply: " << std::endl;
	for (int j = 0; j < row * col; j++)
	{
		v1[j] = 10 + (float)(rand()) / RAND_MAX * (10000 - 10);
	}

	for (int k = 0; k < col; k++)
	{
		v2[k] = 10 + (float)(rand()) / RAND_MAX * (10000 - 10);
	}

	float *result;
	clock_t t6 = clock();
	for (int cnt = 0; cnt < compute_times; cnt++)
	{
		result = scalarMultiply();
	}
	clock_t t7 = clock();
	for (int j = 0; j < row; j++)
	{
		std::cout << "result[" << j << "]: " << result[j] << std::endl;
	}
	std::cout << "==================Original time: " << (float)(t7 - t6) * 1000 / CLOCKS_PER_SEC << std::endl;

	float *avx2_result;
	clock_t t8 = clock();
	for (int cnt = 0; cnt < compute_times; cnt++)
	{
		avx2_result = avxMultiply();
	}
	clock_t t9 = clock();
	for (int j = 0; j < row; j++)
	{
		std::cout << "avx2_result[" << j << "]: " << avx2_result[j] << std::endl;
	}

	std::cout << "==================AVX2 accelerated: " << (float)(t9 - t8) * 1000 / CLOCKS_PER_SEC << std::endl;

	float *avx512_result;
	clock_t t10 = clock();
	for (int cnt = 0; cnt < compute_times; cnt++)
	{
		avx512_result = avx512Multiply();
	}
	clock_t t11 = clock();

	for (int j = 0; j < row; j++)
	{
		std::cout << "avx512_result[" << j << "]: " << avx512_result[j] << std::endl;
	}

	std::cout << "==================AVX512 accelerated: " << (float)(t11 - t10) * 1000 / CLOCKS_PER_SEC << std::endl;

	return 0;
}