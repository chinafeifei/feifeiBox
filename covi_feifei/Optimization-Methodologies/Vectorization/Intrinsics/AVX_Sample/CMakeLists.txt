cmake_minimum_required (VERSION 3.4)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")

add_compile_options(-std=c++17 -mavx512f -mavx2 -mfma -O)

project (avx_sample)

ADD_EXECUTABLE(avx_avg  avx_avg.cpp)

ADD_EXECUTABLE(avx_multiply  avx_multiply.cpp)

ADD_EXECUTABLE(avx_mandelbrot  avx_mandelbrot.cpp)