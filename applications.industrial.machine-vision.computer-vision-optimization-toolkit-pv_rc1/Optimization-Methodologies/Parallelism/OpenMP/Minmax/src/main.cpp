//==============================================================
/* Copyright(C) 2022 Intel Corporation
 * Licensed under the Intel Proprietary License
 */
// =============================================================
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <omp.h>
#include <stdio.h>
#include <chrono>
#include <windows.h>

using namespace std;
using namespace std::chrono;

struct Point
{
	float x;
	float y;
	float z;
};

// create txt file
void createFile();

// Read the txt file
vector<Point> readTxt(const char *);

// Single thread raw_version
void getMinMaxValue(vector<Point> &, int, int, int, float *);

// openMP raw_version2
void getMinMaxValue_openMP2_thread_chunk(vector<Point> &, int, int, int, float *);

// openMP raw_version2
void getMinMaxValue_openMP2_thread_chunk_false_optimal(vector<Point> &, int, int, int, float *);

// openMP raw_version2
void getMinMaxValue_openMP2_thread_chunk_false(vector<Point> &, int, int, int, float *);

// openMP raw_version2
void getMinMaxValue_openMP2_thread_chunk_false_bad(vector<Point> &, int, int, int, float *);

// openMP Version3 with dynamic
void getMinMaxValue_openMP3_thread_chunk(vector<Point> &, int, int, int, float *);

// Print the run time imformation
void cout_info(float, float, float, float, float *);

// The whole test function
void test_fun(int, int, int, vector<Point> &, void (*pf)(vector<Point> &, int, int, int, float *));

int main()
{
	int NUM_THREAD = 8;
	int FOR_TIME = 100;
	int CHUNK = 10000;
	int POINT_NUM = 10100000;

	cout << "The main ThreadId is: " << GetCurrentThreadId() << endl;
	vector<Point> matrixAll;
	// for (int i = 0; i < POINT_NUM; i++) {
	//	Point tmp{};
	//	tmp.x = rand() / double(RAND_MAX);
	//	tmp.y = rand() / double(RAND_MAX);
	//	tmp.z = rand() / double(RAND_MAX);

	//	matrixAll.push_back(tmp);
	//}

	double x_value = 0.01;
	double y_value = 1.01;
	double z_value = 2.01;

	for (int i = 0; i < POINT_NUM; i++)
	{
		Point tmp{};
		x_value += 0.0000001;
		y_value += 0.0000001;
		z_value += 0.0000001;

		tmp.x = x_value;
		tmp.y = y_value;
		tmp.z = z_value;

		matrixAll.push_back(tmp);
	}

	// vector<Point> matrixAll_read = readTxt("YOUR TXT FILE HERE");

	// test_fun(FOR_TIME, NUM_THREAD, CHUNK, matrixAll, getMinMaxValue);
	// Sleep(0.5 * 1000);

	// test_fun(FOR_TIME, NUM_THREAD, CHUNK, matrixAll, getMinMaxValue_openMP3_thread_chunk);
	// Sleep(0.5 * 1000);

	// test_fun(FOR_TIME, NUM_THREAD, CHUNK, matrixAll, getMinMaxValue_openMP2_thread_chunk);
	// Sleep(0.2 * 1000);

	test_fun(FOR_TIME, NUM_THREAD, CHUNK, matrixAll, getMinMaxValue_openMP2_thread_chunk_false_optimal);
	Sleep(0.2 * 1000);

	// test_fun(FOR_TIME, NUM_THREAD, CHUNK, matrixAll, getMinMaxValue_openMP2_thread_chunk_false);
	// Sleep(0.2 * 1000);

	test_fun(FOR_TIME, NUM_THREAD, CHUNK, matrixAll, getMinMaxValue_openMP2_thread_chunk_false_bad);
	Sleep(0.2 * 1000);

	return 0;
}

void createFile()
{
	vector<string> vec;
	for (int i = 0; i < 10100000; i++)
	{
		double input1 = rand() / double(RAND_MAX);
		double input2 = rand() / double(RAND_MAX);
		double input3 = rand() / double(RAND_MAX);
		cout << setiosflags(ios::fixed);
		cout.precision(8);
		std::stringstream s1;
		s1 << fixed << std::setprecision(8) << input1;
		std::stringstream s2;
		s2 << fixed << std::setprecision(8) << input2;
		std::stringstream s3;
		s3 << fixed << std::setprecision(8) << input3;
		string str = s1.str() + " " + s2.str() + " " + s3.str();
		// cout << "str = " << str << endl;
		vec.push_back(str);
	}
	ofstream fout;
	fout.open("test.txt", ios_base::out);
	int len = vec.size();
	for (int i = 0; i < len; ++i)
	{
		fout << vec[i] << endl;
	}
	cout << "txt file created!" << endl;
}

vector<Point> readTxt(const char *fileName)
{
	vector<Point> matrixALL{};

	ifstream fileStream;
	string tmp;
	int count = 0;
	fileStream.open(fileName, ios::in);
	if (fileStream.fail())
	{
		throw std::logic_error("read file fail");
	}
	else
	{
		while (getline(fileStream, tmp, '\n'))
		{
			Point tmpV{};
			string str_tmp_1, str_tmp_2, str_tmp_3;
			istringstream is(tmp);
			is >> str_tmp_1 >> str_tmp_2 >> str_tmp_3;
			tmpV.x = stod(str_tmp_1);
			tmpV.y = stod(str_tmp_2);
			tmpV.z = stod(str_tmp_3);

			matrixALL.push_back(tmpV);
			count++; // Lines
		}
		cout << "Count : " << count << endl;
		fileStream.close();
	}
	return matrixALL;
}

void getMinMaxValue(vector<Point> &mat, int rows, int thread, int chunk, float ans[6])
{
	float max_x_element = mat[0].x;
	float max_y_element = mat[0].y;
	float max_z_element = mat[0].z;

	float min_x_element = mat[0].x;
	float min_y_element = mat[0].y;
	float min_z_element = mat[0].z;

	for (int j = 1; j < rows; j++)
	{
		if (mat[j].x > max_x_element)
			max_x_element = mat[j].x;
		if (mat[j].y > max_y_element)
			max_y_element = mat[j].y;
		if (mat[j].z > max_z_element)
			max_z_element = mat[j].z;

		if (mat[j].x < min_x_element)
			min_x_element = mat[j].x;
		if (mat[j].y < min_y_element)
			min_y_element = mat[j].y;
		if (mat[j].z < min_z_element)
			min_z_element = mat[j].z;
	}

	ans[0] = min_x_element;
	ans[1] = max_x_element;

	ans[2] = min_y_element;
	ans[3] = max_y_element;

	ans[4] = min_z_element;
	ans[5] = max_z_element;
}

void getMinMaxValue_openMP2_thread_chunk(vector<Point> &mat, int rows, int thread, int chunk, float ans[6])
{
	int coreNum = omp_get_num_procs();
	float matrixMinmax[8][16];

	for (int i = 0; i < coreNum; i++)
	{
		matrixMinmax[i][0] = mat[0].x;
		matrixMinmax[i][1] = mat[0].y;
		matrixMinmax[i][2] = mat[0].z;

		matrixMinmax[i][3] = mat[0].x;
		matrixMinmax[i][4] = mat[0].y;
		matrixMinmax[i][5] = mat[0].z;
	}

#pragma omp parallel for num_threads(thread) schedule(dynamic, 10000)
	for (int j = 0; j < rows; j++)
	{
		int k = omp_get_thread_num();
		if (mat[j].x > matrixMinmax[k][0])
			matrixMinmax[k][0] = mat[j].x;
		if (mat[j].y > matrixMinmax[k][1])
			matrixMinmax[k][1] = mat[j].y;
		if (mat[j].z > matrixMinmax[k][2])
			matrixMinmax[k][2] = mat[j].z;

		if (mat[j].x < matrixMinmax[k][3])
			matrixMinmax[k][3] = mat[j].x;
		if (mat[j].y < matrixMinmax[k][4])
			matrixMinmax[k][4] = mat[j].y;
		if (mat[j].z < matrixMinmax[k][5])
			matrixMinmax[k][5] = mat[j].z;
	}

	float max_x_element = matrixMinmax[0][0];
	float max_y_element = matrixMinmax[0][1];
	float max_z_element = matrixMinmax[0][2];

	float min_x_element = matrixMinmax[0][3];
	float min_y_element = matrixMinmax[0][4];
	float min_z_element = matrixMinmax[0][5];

	for (int i = 0; i < coreNum; i++)
	{
		if (matrixMinmax[i][0] > max_x_element)
			max_x_element = matrixMinmax[i][0];
		if (matrixMinmax[i][1] > max_y_element)
			max_y_element = matrixMinmax[i][1];
		if (matrixMinmax[i][2] > max_z_element)
			max_z_element = matrixMinmax[i][2];

		if (matrixMinmax[i][3] < min_x_element)
			min_x_element = matrixMinmax[i][3];
		if (matrixMinmax[i][4] < min_y_element)
			min_y_element = matrixMinmax[i][4];
		if (matrixMinmax[i][5] < min_z_element)
			min_z_element = matrixMinmax[i][5];
	}

	ans[0] = min_x_element;
	ans[1] = max_x_element;

	ans[2] = min_y_element;
	ans[3] = max_y_element;

	ans[4] = min_z_element;
	ans[5] = max_z_element;
}

void getMinMaxValue_openMP2_thread_chunk_false_optimal(vector<Point> &mat, int rows, int thread, int chunk, float ans[6])
{
	int coreNum = omp_get_num_procs();
	float matrixMin[8][16];
	float matrixMax[8][16];

	for (int i = 0; i < coreNum; i++)
	{
		matrixMin[i][0] = mat[0].x;
		matrixMin[i][1] = mat[0].y;
		matrixMin[i][2] = mat[0].z;

		matrixMax[i][0] = mat[0].x;
		matrixMax[i][1] = mat[0].y;
		matrixMax[i][2] = mat[0].z;
	}

#pragma omp parallel for num_threads(thread) schedule(dynamic, 10000)
	for (int j = 0; j < rows; j++)
	{
		int k = omp_get_thread_num();
		if (mat[j].x > matrixMax[k][0])
			matrixMax[k][0] = mat[j].x;
		if (mat[j].y > matrixMax[k][1])
			matrixMax[k][1] = mat[j].y;
		if (mat[j].z > matrixMax[k][2])
			matrixMax[k][2] = mat[j].z;

		if (mat[j].x < matrixMin[k][0])
			matrixMin[k][0] = mat[j].x;
		if (mat[j].y < matrixMin[k][1])
			matrixMin[k][1] = mat[j].y;
		if (mat[j].z < matrixMin[k][2])
			matrixMin[k][2] = mat[j].z;
	}

	float max_x_element = matrixMax[0][0];
	float max_y_element = matrixMax[0][1];
	float max_z_element = matrixMax[0][2];

	float min_x_element = matrixMin[0][0];
	float min_y_element = matrixMin[0][1];
	float min_z_element = matrixMin[0][2];

	for (int i = 0; i < coreNum; i++)
	{
		if (matrixMax[i][0] > max_x_element)
			max_x_element = matrixMax[i][0];
		if (matrixMax[i][1] > max_y_element)
			max_y_element = matrixMax[i][1];
		if (matrixMax[i][2] > max_z_element)
			max_z_element = matrixMax[i][2];

		if (matrixMin[i][0] < min_x_element)
			min_x_element = matrixMin[i][0];
		if (matrixMin[i][2] < min_y_element)
			min_y_element = matrixMin[i][1];
		if (matrixMin[i][1] < min_z_element)
			min_z_element = matrixMin[i][2];
	}

	ans[0] = min_x_element;
	ans[1] = max_x_element;

	ans[2] = min_y_element;
	ans[3] = max_y_element;

	ans[4] = min_z_element;
	ans[5] = max_z_element;
}

void getMinMaxValue_openMP2_thread_chunk_false(vector<Point> &mat, int rows, int thread, int chunk, float ans[6])
{
	int coreNum = omp_get_num_procs();
	float matrixMin[8][3];
	float matrixMax[8][3];

	for (int i = 0; i < coreNum; i++)
	{
		matrixMin[i][0] = mat[0].x;
		matrixMin[i][1] = mat[0].y;
		matrixMin[i][2] = mat[0].z;

		matrixMax[i][0] = mat[0].x;
		matrixMax[i][1] = mat[0].y;
		matrixMax[i][2] = mat[0].z;
	}

#pragma omp parallel for num_threads(thread) schedule(dynamic, 10000)
	for (int j = 0; j < rows; j++)
	{
		int k = omp_get_thread_num();
		if (mat[j].x > matrixMax[k][0])
			matrixMax[k][0] = mat[j].x;
		if (mat[j].y > matrixMax[k][1])
			matrixMax[k][1] = mat[j].y;
		if (mat[j].z > matrixMax[k][2])
			matrixMax[k][2] = mat[j].z;

		if (mat[j].x < matrixMin[k][0])
			matrixMin[k][0] = mat[j].x;
		if (mat[j].y < matrixMin[k][1])
			matrixMin[k][1] = mat[j].y;
		if (mat[j].z < matrixMin[k][2])
			matrixMin[k][2] = mat[j].z;
	}

	float max_x_element = matrixMax[0][0];
	float max_y_element = matrixMax[0][1];
	float max_z_element = matrixMax[0][2];

	float min_x_element = matrixMin[0][0];
	float min_y_element = matrixMin[0][1];
	float min_z_element = matrixMin[0][2];

	for (int i = 0; i < coreNum; i++)
	{
		if (matrixMax[i][0] > max_x_element)
			max_x_element = matrixMax[i][0];
		if (matrixMax[i][1] > max_y_element)
			max_y_element = matrixMax[i][1];
		if (matrixMax[i][2] > max_z_element)
			max_z_element = matrixMax[i][2];

		if (matrixMin[i][0] < min_x_element)
			min_x_element = matrixMin[i][0];
		if (matrixMin[i][2] < min_y_element)
			min_y_element = matrixMin[i][1];
		if (matrixMin[i][1] < min_z_element)
			min_z_element = matrixMin[i][2];
	}

	ans[0] = min_x_element;
	ans[1] = max_x_element;

	ans[2] = min_y_element;
	ans[3] = max_y_element;

	ans[4] = min_z_element;
	ans[5] = max_z_element;
}

void getMinMaxValue_openMP2_thread_chunk_false_bad(vector<Point> &mat, int rows, int thread, int chunk, float ans[6])
{
	int coreNum = omp_get_num_procs();
	float matrixMin_x[8];
	float matrixMin_y[8];
	float matrixMin_z[8];

	float matrixMax_x[8];
	float matrixMax_y[8];
	float matrixMax_z[8];

	for (int i = 0; i < coreNum; i++)
	{
		matrixMin_x[i] = mat[0].x;
		matrixMin_y[i] = mat[0].y;
		matrixMin_z[i] = mat[0].z;

		matrixMax_x[i] = mat[0].x;
		matrixMax_y[i] = mat[0].y;
		matrixMax_z[i] = mat[0].z;
	}

#pragma omp parallel for num_threads(thread) schedule(dynamic, 10000)
	for (int j = 0; j < rows; j++)
	{
		int k = omp_get_thread_num();
		if (mat[j].x > matrixMax_x[k])
			matrixMax_x[k] = mat[j].x;
		if (mat[j].y > matrixMax_y[k])
			matrixMax_y[k] = mat[j].y;
		if (mat[j].z > matrixMax_z[k])
			matrixMax_z[k] = mat[j].z;

		if (mat[j].x < matrixMin_x[k])
			matrixMin_x[k] = mat[j].x;
		if (mat[j].y < matrixMin_x[k])
			matrixMin_y[k] = mat[j].y;
		if (mat[j].z < matrixMin_x[k])
			matrixMin_z[k] = mat[j].z;
	}

	float max_x_element = matrixMax_x[0];
	float max_y_element = matrixMax_y[0];
	float max_z_element = matrixMax_z[0];

	float min_x_element = matrixMin_x[0];
	float min_y_element = matrixMin_y[0];
	float min_z_element = matrixMin_z[0];

	for (int i = 0; i < coreNum; i++)
	{
		if (matrixMax_x[i] > max_x_element)
			max_x_element = matrixMax_x[i];
		if (matrixMax_y[i] > max_y_element)
			max_y_element = matrixMax_y[i];
		if (matrixMax_z[i] > max_z_element)
			max_z_element = matrixMax_z[i];

		if (matrixMin_x[i] < min_x_element)
			min_x_element = matrixMin_x[i];
		if (matrixMin_y[i] < min_y_element)
			min_y_element = matrixMin_y[i];
		if (matrixMin_z[i] < min_z_element)
			min_z_element = matrixMin_z[i];
	}

	ans[0] = min_x_element;
	ans[1] = max_x_element;

	ans[2] = min_y_element;
	ans[3] = max_y_element;

	ans[4] = min_z_element;
	ans[5] = max_z_element;
}

void getMinMaxValue_openMP3_thread_chunk(vector<Point> &mat, int rows, int thread, int chunk, float ans[6])
{
	float max_x_element = mat[0].x;
	float max_y_element = mat[0].y;
	float max_z_element = mat[0].z;

	float min_x_element = mat[0].x;
	float min_y_element = mat[0].y;
	float min_z_element = mat[0].z;

#pragma omp parallel for num_threads(thread) schedule(dynamic, 10000) reduction(max                                                          \
																				: max_x_element, max_y_element, max_z_element) reduction(min \
																																		 : min_x_element, min_y_element, min_z_element)
	for (int j = 1; j < rows; j++)
	{
		if (mat[j].x > max_x_element)
			max_x_element = mat[j].x;
		if (mat[j].y > max_y_element)
			max_y_element = mat[j].y;
		if (mat[j].z > max_z_element)
			max_z_element = mat[j].z;

		if (mat[j].x < min_x_element)
			min_x_element = mat[j].x;
		if (mat[j].y < min_y_element)
			min_y_element = mat[j].y;
		if (mat[j].z < min_z_element)
			min_z_element = mat[j].z;
	}

	ans[0] = min_x_element;
	ans[1] = max_x_element;

	ans[2] = min_y_element;
	ans[3] = max_y_element;

	ans[4] = min_z_element;
	ans[5] = max_z_element;
}

void cout_info(float avg_time, float min_time, float max_time, float varience, float ans[6])
{
	cout << "Avg time is : " << setprecision(3) << avg_time << " ms   ";
	cout << "Min time is : " << setprecision(3) << min_time << " ms   ";
	cout << "Max time is : " << setprecision(3) << max_time << " ms   ";
	cout << "Varience is : " << setprecision(3) << varience << endl;

	cout << "min_value in x column = " << setprecision(3) << fixed << ans[0] << "   ";
	cout << "max_value in x column = " << setprecision(3) << fixed << ans[1] << endl;

	cout << "min_value in y column = " << setprecision(3) << fixed << ans[2] << "   ";
	cout << "max_value in y column = " << setprecision(3) << fixed << ans[3] << endl;

	cout << "min_value in z column = " << setprecision(3) << fixed << ans[4] << "   ";
	cout << "max_value in z column = " << setprecision(3) << fixed << ans[5] << endl;
	cout << "*******************************************" << endl;
}

void test_fun(int FOR_TIME, int NUM_THREAD, int CHUNK, vector<Point> &matrixAll, void (*pf)(vector<Point> &, int, int, int, float *))
{
	float ans[6];

	float avg_time = 0.0;
	float min_time = 1000.0;
	float max_time = 0.0;
	float time_elapsed[10000];
	float variance = 0.0;

	for (int i = 0; i < FOR_TIME; i++)
	{
		auto start = system_clock::now();
		(*pf)(matrixAll, matrixAll.size(), NUM_THREAD, CHUNK, ans);
		auto end = system_clock::now();
		auto dura = duration_cast<microseconds>(end - start);
		float duration = dura.count() / 1000.0f;
		// cout << "The run time is:" << setprecision(4) << time.elapsed_ms() << " ms" << endl;
		avg_time += duration;
		time_elapsed[i] = duration;
		if (min_time > duration)
			min_time = duration;
		if (max_time < duration)
			max_time = duration;
	}
	avg_time /= FOR_TIME;
	for (int i = 0; i < FOR_TIME; i++)
	{
		variance += (time_elapsed[i] - avg_time) * (time_elapsed[i] - avg_time);
	}
	cout_info(avg_time, min_time, max_time, variance, ans);
}