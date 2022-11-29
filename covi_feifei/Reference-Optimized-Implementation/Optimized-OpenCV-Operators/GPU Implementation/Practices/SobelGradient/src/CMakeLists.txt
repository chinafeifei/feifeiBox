if (${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    include_directories(
        "C:/Program Files (x86)/Intel/openvino_2021/opencv/include/"
        ${PROJECT_SOURCE_DIR}/src/sobel.h
    )
    set(OpenCV_DIR "C:/Program Files (x86)/Intel/openvino_2021/opencv/cmake")
    set(CMAKE_CXX_FLAGS "-O3 -Wall -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs \"-device tgllp\"")
else()
    include_directories(
        /opt/intel/openvino_2021/opencv/include/
        ${PROJECT_SOURCE_DIR}/src/sobel.h
    )
    set(OpenCV_DIR "/opt/intel/openvino_2021/opencv/cmake")
    set(CMAKE_CXX_FLAGS "-O3 -Wall -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs \"-device tgllp\"")
endif()

# Set default build type to RelWithDebInfo if not specified
# if (NOT CMAKE_BUILD_TYPE)
#     message (STATUS "Default CMAKE_BUILD_TYPE not set. Using Release with Debug Info")
#     set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
#         STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
#         FORCE)
# endif()

find_package(OpenCV REQUIRED)
file(COPY ../data DESTINATION ${CMAKE_BINARY_DIR}/src)
file(GLOB OpenCV_LIBS 
    /opt/intel/openvino_2021/opencv/lib/libopencv_imgproc.so
    /opt/intel/openvino_2021/opencv/lib/libopencv_core.so
    /opt/intel/openvino_2021/opencv/lib/libopencv_imgcodecs.so
    /opt/intel/openvino_2021/opencv/lib/liblopencv_highgui.so
)

add_executable (gradient-magnitude-sum gradient_magnitude_sum.cpp sobel.dp.cpp)
target_link_libraries(gradient-magnitude-sum ${OpenCV_LIBS} sycl)
add_custom_target (run-sum ./gradient-magnitude-sum)

add_executable (gradient-magnitude-hypot gradient_magnitude_hypot.cpp sobel.dp.cpp)
target_link_libraries(gradient-magnitude-hypot ${OpenCV_LIBS} sycl)
add_custom_target (run-hypot ./gradient-magnitude-hypot)
