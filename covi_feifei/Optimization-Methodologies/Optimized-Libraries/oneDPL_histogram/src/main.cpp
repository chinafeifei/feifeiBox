//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <CL/sycl.hpp>
#include <random>
#include <iostream>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

// Dense algorithm stores all the bins, even if bin has 0 entries
// input array [4,4,1,0,1,2]
// output [(0,1) (1,2)(2,1)(3,0)(4,2)]
// On the other hand, the sparse algorithm excludes the zero-bin values
// i.e., for the sparse algorithm, the same input will give the following output
// [(0,1) (1,2)(2,1)(4,2)]
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
// #include <sys/time.h>

using namespace cv;
using namespace std::chrono;

void dense_histogram(std::vector<uint16_t> &input)
{
  const int N = input.size();
  cl::sycl::buffer<uint16_t, 1> histogram_buf{input.data(),
                                              cl::sycl::range<1>(N)};
  auto start = system_clock::now();

  // Combine the equal values together
  std::sort(oneapi::dpl::execution::dpcpp_default,
            oneapi::dpl::begin(histogram_buf), oneapi::dpl::end(histogram_buf));

  // num_bins is maximum value + 1
  int num_bins;
  {
    sycl::host_accessor histogram(histogram_buf, sycl::read_only);
    num_bins = histogram[N - 1] + 1;

    std::cout << "num bins: " << num_bins << std::endl;
  }
  // sycl::stream
  cl::sycl::buffer<uint16_t, 1> histogram_new_buf{cl::sycl::range<1>(num_bins)};
  cl::sycl::buffer<uint16_t, 1> bins{cl::sycl::range<1>(num_bins)};
  auto val_begin = oneapi::dpl::counting_iterator<int>{0};

  // Determine the end of each bin of value
  oneapi::dpl::upper_bound(
      oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(histogram_buf),
      oneapi::dpl::end(histogram_buf), val_begin, val_begin + num_bins,
      oneapi::dpl::begin(histogram_new_buf));

  // Compute histogram by calculating differences of cumulative histogram
  std::adjacent_difference(oneapi::dpl::execution::dpcpp_default,
                           oneapi::dpl::begin(histogram_new_buf),
                           oneapi::dpl::end(histogram_new_buf),
                           oneapi::dpl::begin(bins));
  auto end = system_clock::now();
  auto duration = duration_cast<milliseconds>(end - start);
  std::cout << "dense_histogram:" << duration.count() * 1.0 << " ms" << std::endl;

  std::cout << "Dense Histogram:\n";
  {
    sycl::host_accessor histogram_new(bins, sycl::read_only);
    std::cout << "[";
    for (int i = 0; i < num_bins; i++)
    {
      std::cout << "(" << i << ", " << histogram_new[i] << ") ";
    }
    std::cout << "]\n";
  }
}

void sparse_histogram(std::vector<uint16_t> &input)
{
  const int N = input.size();
  cl::sycl::buffer<uint16_t, 1> histogram_buf{input.data(),
                                              cl::sycl::range<1>(N)};
  auto start = system_clock::now();

  // Combine the equal values together
  std::sort(oneapi::dpl::execution::dpcpp_default,
            oneapi::dpl::begin(histogram_buf), oneapi::dpl::end(histogram_buf));

  auto num_bins = std::transform_reduce(
      oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(histogram_buf),
      oneapi::dpl::end(histogram_buf), oneapi::dpl::begin(histogram_buf) + 1, 1,
      std::plus<int>(), std::not_equal_to<int>());

  // Create new buffer to store the unique values and their count
  cl::sycl::buffer<uint16_t, 1> histogram_values_buf{
      cl::sycl::range<1>(num_bins)};
  cl::sycl::buffer<uint16_t, 1> histogram_counts_buf{
      cl::sycl::range<1>(num_bins)};

  cl::sycl::buffer<uint16_t, 1> _const_buf{cl::sycl::range<1>(N)};
  std::fill(oneapi::dpl::execution::dpcpp_default,
            oneapi::dpl::begin(_const_buf), oneapi::dpl::end(_const_buf), 1);

  // Find the count of each value
  oneapi::dpl::reduce_by_segment(
      oneapi::dpl::execution::dpcpp_default, oneapi::dpl::begin(histogram_buf),
      oneapi::dpl::end(histogram_buf), oneapi::dpl::begin(_const_buf),
      oneapi::dpl::begin(histogram_values_buf),
      oneapi::dpl::begin(histogram_counts_buf));
  auto end = system_clock::now();
  auto duration = duration_cast<milliseconds>(end - start);
  std::cout << "sparse_histogram:" << duration.count() * 1.0 << " ms" << std::endl;

  std::cout << "Sparse Histogram:\n";
  std::cout << "[";
  for (int i = 0; i < num_bins - 1; i++)
  {
    sycl::host_accessor histogram_value(histogram_values_buf, sycl::read_only);
    sycl::host_accessor histogram_count(histogram_counts_buf, sycl::read_only);
    std::cout << "(" << histogram_value[i] << ", " << histogram_count[i]
              << ") ";
  }
  std::cout << "]\n";
}

#define H_HIGHT 1
#define H_WIDTH 1000

void cv_histogram(std::vector<uint16_t> &input)
{
  float hranges[] = {0, 10};
  const float *ranges[] = {hranges};
  int histSize = 10;
  Mat hist;

  Mat img(H_HIGHT, H_WIDTH, CV_16UC1, &input[0]);
  auto start = system_clock::now();
  calcHist(&img, 1, 0, Mat(), // do not use mask
           hist, 1, &histSize, ranges,
           true, // the histogram is uniform
           false);
  auto end = system_clock::now();
  auto duration = duration_cast<microseconds>(end - start);

  std::cout << "cv_histogram:" << duration.count() * 1.0 << " us" << std::endl;
  std::cout << hist;
  std::cout << "\n";
}

int main(void)
{
  const int N = H_WIDTH;
  std::vector<uint16_t> input, dense, sparse, cv_hist;
  srand((unsigned)time(0));
  // initialize the input array with randomly generated values between 0 and 9
  for (int i = 0; i < N; i++)
    input.push_back(rand() % 10);

  // replacing all input entries of "4" with random number between 1 and 3
  // this is to ensure that we have atleast one entry with zero-bin size,
  // which shows the difference between sparse and dense algorithm output
  for (int i = 0; i < N; i++)
    if (input[i] == 4)
      input[i] = rand() % 3;
  std::cout << "Input:\n";
  for (int i = 0; i < N; i++)
    std::cout << input[i] << " ";
  std::cout << "\n";

  cv_hist = input;
  cv_histogram(cv_hist);

  sparse = input;
  sparse_histogram(sparse);

  dense = input;
  dense_histogram(dense);

  return 0;
}
