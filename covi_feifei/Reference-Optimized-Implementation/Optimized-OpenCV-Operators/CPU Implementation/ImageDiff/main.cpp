#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>
#include <iostream>
#include <chrono>
#include "ippcore.h"
#include "ipps.h"
#include "ippi.h"
#include "ippcv.h"
#include "ippcc.h"
#include "ippvm.h"
#include <omp.h>

#define TIMERSTART(tag)  auto tag##_start = std::chrono::high_resolution_clock::now(),tag##_end = tag##_start
#define TIMEREND(tag)  tag##_end =  std::chrono::high_resolution_clock::now()
#define DURATION_s(tag) printf("%s costs %ld s\n",#tag,std::chrono::duration_cast<std::chrono::seconds>(tag##_end - tag##_start).count())
#define DURATION_ms(tag) printf("%s costs %ld ms\n",#tag,std::chrono::duration_cast<std::chrono::milliseconds>(tag##_end - tag##_start).count());
#define DURATION_us(tag) printf("%s costs %ld us\n",#tag,std::chrono::duration_cast<std::chrono::microseconds>(tag##_end - tag##_start).count());
#define DURATION_ns(tag) printf("%s costs %ld ns\n",#tag,std::chrono::duration_cast<std::chrono::nanoseconds>(tag##_end - tag##_start).count());

using namespace std;
using namespace cv;



void test_opencv_absdiff(cv::Mat &img1, cv::Mat &img2, cv::Mat &diffimg)
{
    TIMERSTART(opencv_absdiff);
    cv::absdiff(img1, img2, diffimg);
    TIMEREND(opencv_absdiff);
    DURATION_us(opencv_absdiff);

    cv::imwrite("opencv_absdiff.png", diffimg);
}


void test_ipp_absdiff(cv::Mat &img1, cv::Mat &img2, cv::Mat &diffimg)
{
    IppiSize roi = { img1.cols, img2.rows };

    diffimg = img1.clone();
    TIMERSTART(ipp_absdiff);

    IppStatus status = ippiAbsDiff_8u_C3R((Ipp8u*)img1.data, img1.step,
                                            (Ipp8u*)img2.data, img2.step,
                                            (Ipp8u*)diffimg.data, img2.step,
                                            roi);
    TIMEREND(ipp_absdiff);
    DURATION_us(ipp_absdiff);
    cv::imwrite("ipp_absdiff.png", diffimg);
}


void test_ipp_tile_absdiff(cv::Mat &img1, cv::Mat &img2, cv::Mat &diffimg)
{
    int numThreads = omp_get_max_threads();
    printf("OMP max thread %d\n", numThreads);
    diffimg = img1.clone();

    for(int c = numThreads; c >= 2; c = c / 2) {
        printf("OMP %d threads Test: ", c);

        IppiSize roi = { img1.cols/c, img2.rows };
        int step_size = img1.cols/c * img2.rows *3;

        TIMERSTART(ipp_tile_absdiff);
        #pragma omp parallel for
        for(int i=0; i < c; i++){
            IppStatus status = ippiAbsDiff_8u_C3R((Ipp8u*)(img1.data + i * step_size), img1.step/c,
                                                    (Ipp8u*)(img2.data + i * step_size), img2.step/c,
                                                    (Ipp8u*)(diffimg.data + i * step_size), img2.step/c,
                                                    roi);
        }
        TIMEREND(ipp_tile_absdiff);
        DURATION_us(ipp_tile_absdiff);
        std::string filename = "ipp_tile_absdiff_" + std::to_string(c) + ".png";
        cv::imwrite(filename, diffimg);
    }
}

void compare_result(cv::Mat &m1, cv::Mat &m2)
{
    int step = m1.cols * m1.channels();
    int same = 0;
    int diff = 0;
    for (int x = 0; x < m1.rows; x++) {
        for (int y = 0; y < step; y++) {
            if (m1.data[x * step + y] == m2.data[x * step + y]) {
                same++;
            }
            else {
                diff++;
            }
        }
    }

    float correctness = same / (same + diff) * 100;

    printf("pixels same is %d, diff is %d, correctness is %2.2f %%\n", same, diff, correctness);
}

void test(cv::Mat img1, cv::Mat img2)
{
    cv::Mat cv_result, ipp_result, ipp_tile_result;

    std::cout << "=================OpenCV absdiff===================" << std::endl;
    test_opencv_absdiff(img1, img2, cv_result);
    std::cout << "=================IPP absdiff======================" << std::endl;
    test_ipp_absdiff(img1, img2, ipp_result);
    compare_result(cv_result, ipp_result);
    std::cout << "=================IPP tile absdiff=================" << std::endl;
    test_ipp_tile_absdiff(img1, img2, ipp_tile_result);
    compare_result(ipp_result, ipp_tile_result);

}

int main(int argc, char* argv[])
{
    cv::Mat img1, img2;
    img1 = cv::imread("../data/frame_050_rs.png");
    img2 = cv::imread("../data/frame_080_rs.png");
    std::cout << "Frame size: " << img1.rows << " * " << img1.cols <<std::endl;

    test(img1, img2);

    cv::Mat img3, img4;
    img3 = cv::imread("../data/frame_050_lg.png");
    img4 = cv::imread("../data/frame_080_lg.png");
    std::cout << "\nFrame size: " << img3.rows << " * " << img4.cols <<std::endl;

    test(img3, img4);

	return 0;
}

