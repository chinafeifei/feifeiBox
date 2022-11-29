cmake_minimum_required(VERSION 3.4)

if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    set(CMAKE_CXX_COMPILER "dpcpp-cl")
else()
    set(CMAKE_CXX_COMPILER "dpcpp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -pthread -fopenmp")
    OPTION(USE_OpenMP "Use OpenMP" ON)
endif()

project(IPP_MULTITHREADING LANGUAGES CXX)
add_subdirectory(src)