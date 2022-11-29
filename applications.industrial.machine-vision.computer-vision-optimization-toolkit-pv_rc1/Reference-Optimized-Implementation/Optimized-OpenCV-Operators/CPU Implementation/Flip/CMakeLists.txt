if (${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    set(IPP_DIR "C:/Program Files (x86)/Intel/oneAPI/ipp/latest/lib/intel64")
    include_directories("C:/Program Files (x86)/Intel/oneAPI/ipp/latest/include/")
    set(OpenCV_DIR "C:/Program Files (x86)/IntelSWTools/openvino_2021/opencv/")
else()
    set(IPP_DIR "/opt/intel/oneapi/ipp/latest/lib/intel64")
    include_directories("/opt/intel/oneapi/ipp/latest/include/")
    set(OpenCV_DIR "/opt/intel/openvino_2021/opencv/")
endif()

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set. Using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

file(COPY ./data DESTINATION ${CMAKE_BINARY_DIR})
find_package(OpenCV REQUIRED)
find_library(IPP_CORE ippcore PATHS ${IPP_DIR})
find_library(IPP_I ippi PATHS ${IPP_DIR})
find_library(IPP_S ipps PATHS ${IPP_DIR})
find_library(IPP_CC ippcc PATHS ${IPP_DIR})

add_executable (Flip src/Flip.cpp)

if (${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    target_link_libraries(Flip ${IPP_CORE} ${IPP_S} ${IPP_I}  ${IPP_CC} ${OpenCV_LIBS})
else()
    target_link_libraries(Flip ${IPP_CORE} ${IPP_S} ${IPP_I}  ${IPP_CC} ${OpenCV_LIBS})
endif()

add_custom_target (run-Flip ./Flip )
