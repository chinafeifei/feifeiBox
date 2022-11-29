if (${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    include_directories(
        "C:/Program Files (x86)/Intel/openvino_2021/opencv/include/"
        ${PROJECT_SOURCE_DIR}/src/erosion.h
        ${PROJECT_SOURCE_DIR}/src/esimd_test_utils.hpp
    )
    set(OpenCV_DIR "C:/Program Files (x86)/Intel/openvino_2021/opencv/cmake")
    set(CMAKE_CXX_FLAGS "-O2 -fsycl")
else()
    include_directories(
        /opt/intel/openvino_2021/opencv/include/
        ${PROJECT_SOURCE_DIR}/src/erosion.h
        ${PROJECT_SOURCE_DIR}/src/esimd_test_utils.hpp
    )
    set(OpenCV_DIR "/opt/intel/openvino_2021/opencv/cmake")
    set(CMAKE_CXX_FLAGS "-O2 -fsycl")
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

add_executable (erosion erosion.cpp erosion.dp.cpp)
target_link_libraries(erosion ${OpenCV_LIBS} sycl)
add_custom_target (run-erosion ./erosion)
