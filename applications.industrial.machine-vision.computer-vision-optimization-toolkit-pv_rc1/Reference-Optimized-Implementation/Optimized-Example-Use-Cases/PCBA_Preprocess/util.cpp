#pragma once
#include "util.h"


void comparecv(cv::Mat& image, cv::Mat& output) {

    float same = 0.0f;
    for (int x = 0; x < image.rows; x++) {
        for (int y = 0; y < image.cols; y++) {

            if (image.at<uchar>(x, y) == output.at<uchar>(x, y)) {

                same++;

            }

        }
    }

    printf("correctness is %.3f %%\n", same / image.total() * 100);

}

void displayArray(cv::Mat& image, bool DEBUG)
{
    if (!DEBUG)
        return;
    std::cout << "img (grad_x) = \n"
        << cv::format(image, cv::Formatter::FMT_C) << ";" << std::endl
        << std::endl
        << std::endl;
}

