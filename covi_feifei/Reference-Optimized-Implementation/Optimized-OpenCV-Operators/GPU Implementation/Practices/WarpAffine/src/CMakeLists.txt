if (${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    include_directories(
        "C:/Program Files (x86)/Intel/openvino_2021/opencv/include/"
        ${PROJECT_SOURCE_DIR}/src/cudaWarp.h
        ${PROJECT_SOURCE_DIR}/src/mat33.h
    )
    set(OpenCV_DIR "C:/Program Files (x86)/Intel/openvino_2021/opencv/cmake")
    set(CMAKE_CXX_FLAGS "-fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs \"-device tgllp\"")
else()
    include_directories(
        /opt/intel/openvino_2021/opencv/include/
        ${PROJECT_SOURCE_DIR}/src/cudaWarp.h
        ${PROJECT_SOURCE_DIR}/src/mat33.h
    )
    set(OpenCV_DIR "/opt/intel/openvino_2021/opencv/cmake")
    set(CMAKE_CXX_FLAGS "-fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs \"-device tgllp\"")
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
)

add_executable (warp-affine warpAffine_transformation.cpp cudaWarp-affine.dp.cpp)
target_link_libraries(warp-affine ${OpenCV_LIBS} sycl)
add_custom_target (run-warp-affine ./warp-affine)
