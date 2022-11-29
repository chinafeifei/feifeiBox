if (${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /Qtbb")
    set(IPP_DIR "C:/Program Files (x86)/Intel/oneAPI/ipp/latest/lib/intel64")
    set(IPP_TL_TBB_DIR "C:/Program Files (x86)/Intel/oneAPI/ipp/latest/lib/intel64/tl/tbb")
    include_directories("C:/Program Files (x86)/Intel/oneAPI/ipp/latest/include/")
    set(OpenCV_DIR "C:/Program Files (x86)/IntelSWTools/openvino_2021/opencv/")
else()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -qtbb")
    set(IPP_DIR "/opt/intel/oneapi/ipp/latest/lib/intel64")
    set(IPP_TL_TBB_DIR "/opt/intel/oneapi/ipp/latest/lib/intel64/tl/tbb")
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

file(COPY ../data DESTINATION ${CMAKE_BINARY_DIR}/src)
find_package(OpenCV REQUIRED)
find_library(IPP_CORE ippcore PATHS ${IPP_DIR})
find_library(IPP_I ippi PATHS ${IPP_DIR})
find_library(IPP_S ipps PATHS ${IPP_DIR})
find_library(IPP_I_TL_TBB ippi_tl_tbb PATHS ${IPP_TL_TBB_DIR})
find_library(IPP_CORE_TL_TBB ippcore_tl_tbb PATHS ${IPP_TL_TBB_DIR})
find_library(IPP_MT_TL_TBB ippimt_tl_tbb PATHS ${IPP_TL_TBB_DIR})



add_executable (gradient-magnitude-sum gradient_magnitude_sum.cpp)
target_link_libraries(gradient-magnitude-sum ${IPP_CORE} ${IPP_S} ${IPP_I} ${IPP_CORE_TL_TBB} ${IPP_I_TL_TBB} ${OpenCV_LIBS})
add_custom_target (run-sum ./gradient-magnitude-sum)
add_executable (gradient-magnitude-hypot gradient_magnitude_hypot.cpp)
if (${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    target_link_libraries(gradient-magnitude-hypot ${IPP_CORE} ${IPP_S} ${IPP_I} ${IPP_CORE_TL_TBB} ${IPP_I_TL_TBB} ${IPP_MT_TL_TBB} ${OpenCV_LIBS})
else()
    target_link_libraries(gradient-magnitude-hypot ${IPP_CORE} ${IPP_S} ${IPP_I} ${IPP_CORE_TL_TBB} ${IPP_I_TL_TBB} ${OpenCV_LIBS})
endif()


add_custom_target (run-hypot ./gradient-magnitude-hypot)
