cmake_minimum_required(VERSION 3.0.0)

project(absdiff_opt VERSION 0.1.0)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -mavx2")

include_directories("/opt/intel/oneapi/ipp/latest/include/") 
set(IPP_DIR "/opt/intel/oneapi/ipp/latest/lib/intel64") 

find_package(OpenCV REQUIRED)
find_library(IPP_CORE ippcore PATHS ${IPP_DIR})
find_library(IPP_I ippi PATHS ${IPP_DIR})
find_library(IPP_S ipps PATHS ${IPP_DIR})
find_library(IPP_CV ippcv PATHS ${IPP_DIR})
find_library(IPP_VM ippvm PATHS ${IPP_DIR})

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable (absdiff_opt main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${IPP_CORE} ${IPP_S} ${IPP_I} ${IPP_VM} ${IPP_CV})

add_custom_target (run ./absdiff_opt)