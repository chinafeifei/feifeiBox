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
#include <windows.h>
#include <thread>
#include <algorithm>
#include <time.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define TH_NUMS 8
float m_value[TH_NUMS][6]; // The max and min value obtained by each thread
const int preemptThreadNum = 4;

struct Point {
	float x; float y; float z;
};

//create txt file
void createFile() {
	vector<string> vec;
	for (int i = 0; i < 10100000; i++) {
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
		//cout << "str = " << str << endl;
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

//read txt file
std::vector<Point> readTxt(const char* fileName) {
	std::vector<Point> matrixALL{};
	int row = 0;

	std::ifstream fileStream;
	std::string tmp;
	int count = 0;
	fileStream.open(fileName, std::ios::in);
	if (fileStream.fail()) {
		throw std::logic_error("read file fail");
	}
	else {
		while (getline(fileStream, tmp, '\n')) {
			Point tmpV{};
			std::istringstream is(tmp);
			std::string str_tmp_1, str_tmp_2, str_tmp_3;
			is >> str_tmp_1 >> str_tmp_2 >> str_tmp_3;
			tmpV.x = std::stof(str_tmp_1);
			tmpV.y = std::stof(str_tmp_2);
			tmpV.z = std::stof(str_tmp_3);

			matrixALL.push_back(tmpV);
			count++;
		}
		cout << "total number of rows : " << count << endl;
		fileStream.close();
	}
	return matrixALL;
}

//Find the max and min values of each column
void getMinMaxValue(std::vector<Point>& mat, int start, int end, float res[6]) {
	float max_x_element = mat[start].x;
	float max_y_element = mat[start].y;
	float max_z_element = mat[start].z;

	float min_x_element = mat[start].x;
	float min_y_element = mat[start].y;
	float min_z_element = mat[start].z;

	for (int j = start; j < end; j++) {
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
	//Put the max and min values into res
	res[0] = min_x_element;
	res[2] = min_y_element;
	res[4] = min_z_element;
	res[1] = max_x_element;
	res[3] = max_y_element;
	res[5] = max_z_element;
}


void otherOperation() {
	int* arr = new int[100000];
	for (int i = 0; i < 100000; i++) {
		arr[i] = rand();
	}
	sort(arr, arr + 100000);
}

void compareValueInM_value(float res[6]) {
	res[0] = INT_MAX;
	res[2] = INT_MAX;
	res[4] = INT_MAX;
	res[1] = INT_MIN;
	res[3] = INT_MIN;
	res[5] = INT_MIN;
	for (int i = 0; i < TH_NUMS; i++) {
		res[0] = min(m_value[i][0], res[0]);
		res[1] = max(m_value[i][1], res[1]);
		res[2] = min(m_value[i][2], res[2]);
		res[3] = max(m_value[i][3], res[3]);
		res[4] = min(m_value[i][4], res[4]);
		res[5] = max(m_value[i][5], res[5]);
	}
}

int main(void) {
	SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
	std::cout << "main thread id: " << GetCurrentThreadId() << endl;

	//createFile();  //if you have no file, use this function to create test.txt file
	Sleep(100);
	std::vector<Point> matrixAll = readTxt("test.txt");//read txt file

	int runNum = 100;//The number of times the function is repeated
	double avetime = 0.0;
	double maxtime = -DBL_MAX;
	double mintime = DBL_MAX;

	cout << "******get core number******  " << endl;
	SYSTEM_INFO systemInfo;
	GetSystemInfo(&systemInfo);
	int coreNumber = systemInfo.dwNumberOfProcessors;
	cout << "coreNumber:" << coreNumber << endl<<endl<<endl;

	cout << "******single thread test******  " << endl;
	float single_thread[6];
	auto start = system_clock::now();
	thread singleThread(getMinMaxValue, std::ref(matrixAll), 0, matrixAll.size(), single_thread);
	singleThread.join();
	auto end = system_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	double curTime = duration.count() / 1000.000;
	//for (int i = 0; i < 3; i++) {
	//	cout << "min value in Column " << i + 1 << " = " << setprecision(8) << fixed << single_thread[i * 2] << "  ,  ";
	//	cout << "max value in Column " << i + 1 << " = " << setprecision(8) << fixed << single_thread[i * 2 + 1] << endl;
	//}
	cout << "The single thread run time is:" << fixed << std::setprecision(3) << curTime << " ms" << endl << endl << endl;

	Sleep(50);


	cout << "******multi thread test******  " << endl;

	thread multiThread[TH_NUMS];
	cout << "create thread numbers = " << TH_NUMS << endl;
	for (int n = 0; n < runNum; n++) {
		//Create threads to preempt CPU resources
		thread cpuPreemptThread[preemptThreadNum];
		for (int i = 0; i < preemptThreadNum; i++) {
			cpuPreemptThread[i] = thread(otherOperation);
			//std::cout << "cpuPreemptThread  " << i << ":" << cpuPreemptThread[i].get_id() << endl;
		}
		auto start = system_clock::now();
		for (int i = 0; i < TH_NUMS; i++) {
			multiThread[i] = thread(getMinMaxValue, std::ref(matrixAll),
				i * matrixAll.size() / TH_NUMS, (i + 1) * matrixAll.size() / TH_NUMS, m_value[i]);
		}
		for (int i = 0; i < TH_NUMS; i++) {
			multiThread[i].join();
		}

		//Compare the results calculated by each thread to find the max and min values
		float res[6]; 
		compareValueInM_value(res);
		auto end = system_clock::now();
		auto duration = duration_cast<microseconds>(end - start);
		double curTime = duration.count() / 1000.000;
		//for (int i = 0; i < 3; i++) {
		//	cout << "min value in Column " << i + 1 << " = " << setprecision(8) << fixed << res[i * 2] << "  ,  ";
		//	cout << "max value in Column " << i + 1 << " = " << setprecision(8) << fixed << res[i * 2 + 1] << endl;
		//}
		cout << "The multi thread run time is:" << fixed << std::setprecision(3) << curTime << " ms" << endl;
		avetime += curTime;
		if (curTime > maxtime)
		{
			maxtime = curTime;
		}
		if (curTime < mintime)
		{
			mintime = curTime;
		}
		for (int i = 0; i < preemptThreadNum; i++) {
			cpuPreemptThread[i].join();
		}
		Sleep(500);
	}
	std::cout << "avetime: " << fixed << std::setprecision(3) << avetime / runNum << std::endl;
	std::cout << "maxtime: " << fixed << std::setprecision(3) << maxtime << std::endl;
	std::cout << "mintime: " << fixed << std::setprecision(3) << mintime << std::endl;

	cout << endl << endl << endl;
	Sleep(50);

	cout << "****** multi thread with thread affinity test******  " << endl;
	//reset
	avetime = 0.0;
	maxtime = -DBL_MAX;
	mintime = DBL_MAX;
	thread multiThreadWithAffinity[TH_NUMS - 1];
	cout << "create thread numbers = " << TH_NUMS - 1 <<endl;
	cout << "And other threads operation bind to a single cpu " << endl;
	for (int n = 0; n < runNum; n++) {
		//Create threads to preempt CPU sources and bind the preemptive thread to run on a single core
		thread cpuPreemptThread_1[preemptThreadNum];
		for (int i = 0; i < preemptThreadNum; i++) {
			cpuPreemptThread_1[i] = thread(otherOperation);
			SetThreadPriority(cpuPreemptThread_1[i].native_handle(), THREAD_PRIORITY_LOWEST);
			SetThreadAffinityMask(cpuPreemptThread_1[i].native_handle(), DWORD_PTR(1) << (TH_NUMS - 1));
		}

		auto start = system_clock::now();
		for (int i = 0; i < TH_NUMS - 1; i++) {
			multiThreadWithAffinity[i] = thread(getMinMaxValue, std::ref(matrixAll),
				i * matrixAll.size() / (TH_NUMS - 1), (i + 1) * matrixAll.size() / (TH_NUMS - 1), m_value[i]);
			//Set Thread Priority
			SetThreadPriority(multiThreadWithAffinity[i].native_handle(), THREAD_PRIORITY_TIME_CRITICAL);
			//Set Thread Affinity
			DWORD_PTR dw = SetThreadAffinityMask(multiThreadWithAffinity[i].native_handle(), DWORD_PTR(1) << i);
			if (dw == 0)
			{
				DWORD dwErr = GetLastError();
				std::cerr << "SetThreadAffinityMask failed, GLE=" << dwErr << '\n';
			}
		}

		for (int i = 0; i < TH_NUMS - 1; i++) {
			multiThreadWithAffinity[i].join();
		}

		//Compare the results calculated by each thread to find the max and min values
		float res[6]; 
		compareValueInM_value(res);
		auto end = system_clock::now();
		auto duration = duration_cast<microseconds>(end - start);
		double curTime = duration.count() / 1000.000;
		//for (int i = 0; i < 3; i++) {
		//	cout << "min value in Column " << i + 1 << " = " << setprecision(8) << fixed << res[i * 2] << "  ,  ";
		//	cout << "max value in Column " << i + 1 << " = " << setprecision(8) << fixed << res[i * 2 + 1] << endl;
		//}
		cout << "The multi thread with thread affinity run time is:" << fixed << std::setprecision(3) << curTime << " ms" << endl;
		avetime += curTime;
		if (curTime > maxtime)
		{
			maxtime = curTime;
		}
		if (curTime < mintime)
		{
			mintime = curTime;
		}

		for (int i = 0; i < preemptThreadNum; i++) {
			cpuPreemptThread_1[i].join();
		}
		Sleep(500);
	}
	std::cout << "avetime: " << fixed << std::setprecision(3) << avetime / runNum << std::endl;
	std::cout << "maxtime: " << fixed << std::setprecision(3) << maxtime << std::endl;
	std::cout << "mintime: " << fixed << std::setprecision(3) << mintime << std::endl;
	Sleep(1000);
	return 0;
}