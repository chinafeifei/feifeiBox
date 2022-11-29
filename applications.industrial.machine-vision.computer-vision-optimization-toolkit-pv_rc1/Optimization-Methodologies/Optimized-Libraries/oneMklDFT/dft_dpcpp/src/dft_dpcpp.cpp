//==============================================================
/* Copyright(C) 2022 Intel Corporation
* Licensed under the Intel Proprietary License
*/
// =============================================================

#pragma once
#include <CL/sycl.hpp>-
#include <stdio.h>
#include <iostream>
#include <oneapi/mkl.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using millis = std::chrono::milliseconds;
using namespace std;
using namespace cv;

static const bool DEBUG = false;
static const bool INFO = true;

typedef oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX> descriptor_t;

// local includes
#define NO_MATRIX_HELPERS
#include "common_for_examples.hpp"

constexpr int SUCCESS = 0;
constexpr int FAILURE = 1;

int opencv_dft_test()
{
    Mat src = imread("../data/input.jpg", IMREAD_GRAYSCALE);

    if (src.empty())
    {
        printf("read image failed! \n");
        return -1;
    }
    else
        printf("read image succeed!\n");

    Mat padded;
    int m = getOptimalDFTSize(src.rows);
    int n = getOptimalDFTSize(src.cols);

    copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(),CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);

    auto start = std::chrono::high_resolution_clock::now();
    dft(complexI, complexI);
    idft(complexI, complexI);

    auto stop = std::chrono::high_resolution_clock::now();
    auto opencv_time = std::chrono::duration_cast<millis>(stop - start).count();

    if (INFO)
        printf("opencv dft/idft took %ld ms \n", opencv_time);

    return 0;
}


int run_dft_example(cl::sycl::device& dev) {
     int result = FAILURE;

    try {
        // Catch asynchronous exceptions
        auto exception_handler = [](cl::sycl::exception_list exceptions) {
            for (std::exception_ptr const& e : exceptions) {
                try {
                    std::rethrow_exception(e);
                }
                catch (cl::sycl::exception const& e) {
                    std::cout << "Caught asynchronous SYCL exception:" << std::endl
                        << e.what() << std::endl;
                }
            }
        };

        // create execution queue with asynchronous error handling
        cl::sycl::queue queue(dev, exception_handler);

        // create execution queue with asynchronous error handling
        //cl::sycl::queue queue(dev, exception_handler);
       // cl::sycl::queue queue{ cl::sycl::cpu_selector{} };

        std::cout << "\noneMKL DFT Running on device: "
            << queue.get_device().get_info<cl::sycl::info::device::name>() << "\n";

        Mat img_in = imread("../data/input.jpg", 0);
        if (img_in.empty()) {
            printf(" read image failed \n");
        }

        img_in.convertTo(img_in, CV_32F, 1.0 / 255, 0);

        const int  width = img_in.cols;
        const int  height = img_in.rows;
        int N1 = height, N2 = width;

        // Setting up USM and initialization
        float* in_usm = (float*)malloc_shared(N2 * N1 * 2 * sizeof(float), queue.get_device(), queue.get_context());
        float* in_usm1 = (float*)malloc_shared(N2 * N1 * 2 * sizeof(float), queue.get_device(), queue.get_context());
        float* out_usm = (float*)malloc_shared(N2 * N1 * 2 * sizeof(float), queue.get_device(), queue.get_context());
        float* out_usm1 = (float*)malloc_shared(N2 * N1 * 2 * sizeof(float), queue.get_device(), queue.get_context());

        for (int i = 0; i < height; ++i)
        {
            float* imgPtr = img_in.ptr<float>(i);
            for (int j = 0; j < width; ++j)
            {
                in_usm[i * width + j] = (float)imgPtr[j];
                in_usm1[i * width + j] = (float)imgPtr[j];
            }
        }

        {
            
            descriptor_t desc({ N2, N1 });
            desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0 / (N1 * N2)));
            desc.commit(queue);

            cl::sycl::event fwd, bwd;
            fwd = oneapi::mkl::dft::compute_forward(desc, in_usm);
            fwd.wait();
            bwd = oneapi::mkl::dft::compute_backward(desc, in_usm);
            bwd.wait();

            auto start = std::chrono::high_resolution_clock::now();
            fwd = oneapi::mkl::dft::compute_forward(desc, in_usm);
            fwd.wait();
            bwd = oneapi::mkl::dft::compute_backward(desc, in_usm);
            bwd.wait();
            auto stop = std::chrono::high_resolution_clock::now();
            auto time = std::chrono::duration_cast<millis>(stop - start).count();

            if (INFO)
                printf("oneMKL dpcpp  in-place dft/idft took %ld ms \n", time);
        }

 
        {
            descriptor_t desc_dft({ N2, N1 });
            desc_dft.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
            desc_dft.commit(queue);

            sycl::event ifft2d_ev, ifft2d_ev1;

            ifft2d_ev = oneapi::mkl::dft::compute_forward(
                desc_dft,
                (float*)in_usm1,
                (float*)out_usm);
            ifft2d_ev.wait();
            ifft2d_ev1 = oneapi::mkl::dft::compute_backward(
                desc_dft,
                (float*)out_usm,
                (float*)out_usm1);
            ifft2d_ev1.wait();

            auto start = std::chrono::high_resolution_clock::now();
            ifft2d_ev = oneapi::mkl::dft::compute_forward(
                desc_dft,
                (float*)in_usm1,
                (float*)out_usm);
            ifft2d_ev.wait();
            ifft2d_ev1 = oneapi::mkl::dft::compute_backward(
                desc_dft,
                (float*)out_usm,
                (float*)out_usm1);
            ifft2d_ev1.wait();

            auto stop = std::chrono::high_resolution_clock::now();
            auto time = std::chrono::duration_cast<millis>(stop - start).count();

            if (INFO)
                printf("oneMKL dpcpp out-place dft/idft took %ld ms \n", time);
        }

        result = SUCCESS;
        free(in_usm, queue.get_context());
        free(in_usm1, queue.get_context());
        free(out_usm, queue.get_context());
        free(out_usm1, queue.get_context());
    }
    catch (cl::sycl::exception const& e) {
        std::cout << "\t\tSYCL exception during FFT" << std::endl;
        std::cout << "\t\t" << e.what() << std::endl;
        std::cout << "\t\tOpenCl status: " << e.get_cl_code() << std::endl;
    }
    catch (std::runtime_error const& e) {
        std::cout << "\t\truntime exception during FFT" << std::endl;
        std::cout << "\t\t" << e.what() << std::endl;
    }

    return result;
}
//
// Description of example setup, apis used and supported floating point type precisions
//
void print_example_banner() {
    std::cout << "" << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << "# 2D FFT Complex-Complex Single-Precision Example: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   dft" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Supported floating point type precisions:" << std::endl;
    std::cout << "#   float" << std::endl;
    std::cout << "#   std::complex<float>" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << std::endl;
}

//
// Main entry point for example.
//
// Dispatches to appropriate device types as set at build time with flag:
// -DSYCL_DEVICES_host -- only runs host implementation
// -DSYCL_DEVICES_cpu -- only runs SYCL CPU implementation
// -DSYCL_DEVICES_gpu -- only runs SYCL GPU implementation
// -DSYCL_DEVICES_all (default) -- runs on all: host, cpu and gpu devices
//
//  For each device selected and each supported data type, Basic_Sp_C2C_2D_FFTExample
//  runs is with all supported data types
//
int main() {
    print_example_banner();

    opencv_dft_test();

    std::list<my_sycl_device_types> list_of_devices;
    set_list_of_devices(list_of_devices);

    int returnCode = 0;
    for (auto it = list_of_devices.begin(); it != list_of_devices.end(); ++it) {
        cl::sycl::device my_dev;
        bool my_dev_is_found = false;
        get_sycl_device(my_dev, my_dev_is_found, *it);

        if (my_dev_is_found) {
            //std::cout << "Running tests on " << sycl_device_names[*it] << ".\n";
            int status = run_dft_example(my_dev);
            if (status != SUCCESS) {
                std::cout << "\tTest Failed" << std::endl << std::endl;
                returnCode = status;
            }
        }
        else {
#ifdef FAIL_ON_MISSING_DEVICES
            std::cout << "No " << sycl_device_names[*it] << " devices found; Fail on missing devices is enabled." << std::endl;
            return 1;
#else
            std::cout << "No " << sycl_device_names[*it] << " devices found; skipping " << sycl_device_names[*it] << " tests." << std::endl << std::endl;
#endif
        }
    }


    return returnCode;
}