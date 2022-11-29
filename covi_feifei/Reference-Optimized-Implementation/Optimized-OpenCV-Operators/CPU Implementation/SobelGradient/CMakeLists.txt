cmake_minimum_required (VERSION 3.4)

if (${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    set(CMAKE_CXX_COMPILER "dpcpp-cl")
else()
    set(CMAKE_CXX_COMPILER "dpcpp")
endif()

project (SOBEL_WIPP_TBB LANGUAGES CXX)
add_subdirectory (src)
