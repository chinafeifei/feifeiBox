if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /Qtbb")
    set(IPP_LIB_DIR "C:/Program Files (x86)/Intel/oneAPI/ipp/latest/lib/intel64")
    set(TBB_LIB_DIR "C:/Program Files (x86)/Intel/oneAPI/tbb/latest/lib/intel64/vc14")
    include_directories("C:/Program Files (x86)/Intel/oneAPI/ipp/latest/include")
    include_directories("C:/Program Files (x86)/Intel/oneAPI/tbb/latest/include")
    set(OpenCV_DIR "C:/Program Files (x86)/IntelSWTools/openvino_2021/opencv/")
else()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -qtbb")
    set(IPP_DIR "/opt/intel/oneapi/ipp/latest/lib/intel64")
    set(IPP_TL_TBB_DIR "/opt/intel/oneapi/ipp/latest/lib/intel64/tl/tbb")
    include_directories("/opt/intel/oneapi/ipp/latest/include/")
    set(OpenCV_DIR "/opt/intel/openvino_2021/opencv/")
    set(ITT_DIR "/opt/intel/oneapi/vtune/latest/lib64")
    include_directories("/opt/intel/oneapi/vtune/latest/include")
endif()

# Set default build type to RelWithDebInfo if not specified
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Default CMAKE_BUILD_TYPE not set. Using Release with Debug Info")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

file(COPY ../data DESTINATION ${CMAKE_BINARY_DIR}/src)
find_package(OpenCV REQUIRED)
find_library(IPP_CORE ippcore PATHS ${IPP_DIR})
find_library(IPP_I ippi PATHS ${IPP_DIR})
find_library(IPP_S ipps PATHS ${IPP_DIR})
find_library(IPP_CV ippcv PATHS ${IPP_DIR})
find_library(IPP_I_TL_TBB ippi_tl_tbb PATHS ${IPP_TL_TBB_DIR})
find_library(IPP_CORE_TL_TBB ippcore_tl_tbb PATHS ${IPP_TL_TBB_DIR})
find_library(IPP_MT_TL_TBB ippimt_tl_tbb PATHS ${IPP_TL_TBB_DIR})
find_library(IPP_IW1 libipp_iw.a PATHS ${IPP_DIR})
find_library(IPP_IW2 libippcc.so PATHS ${IPP_DIR})

# find_library(IPP_IW3 libippcv.a  PATHS ${IPP_DIR})
# find_library(IPP_IW4 libippi.a PATHS ${IPP_DIR})
# find_library(IPP_IW5 libipps.a  PATHS ${IPP_DIR})
find_library(IPP_IW6 libippvm.so PATHS ${IPP_DIR})
find_library(ITT_NOTIFY ittnotify PATHS ${ITT_DIR})

add_definitions(-D__TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION_PRESENT)
add_executable(mean-blur11-avx2-fp16-reg mean-blur11-avx2-fp16-reg.cpp)

if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    target_link_libraries(mean-blur11-avx2-fp16-reg ${IPP_CORE} ${IPP_S} ${IPP_I} ${IPP_CV} ${IPP_CORE_TL_TBB} ${IPP_I_TL_TBB} ${IPP_MT_TL_TBB} ${OpenCV_LIBS})
else()
    target_link_libraries(mean-blur11-avx2-fp16-reg ${IPP_IW1} ${IPP_IW2} ${IPP_CV} ${IPP_I} ${IPP_S} ${IPP_IW6} ${IPP_CORE} ${IPP_CORE_TL_TBB} ${IPP_I_TL_TBB} ${OpenCV_LIBS} ${ITT_NOTIFY})
endif()

add_executable(mean-blur11-avx2-fp16 mean-blur11-avx2-fp16.cpp)

if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    target_link_libraries(mean-blur11-avx2-fp16 ${IPP_CORE} ${IPP_S} ${IPP_I} ${IPP_CV} ${IPP_CORE_TL_TBB} ${IPP_I_TL_TBB} ${IPP_MT_TL_TBB} ${OpenCV_LIBS})
else()
    target_link_libraries(mean-blur11-avx2-fp16 ${IPP_IW1} ${IPP_IW2} ${IPP_CV} ${IPP_I} ${IPP_S} ${IPP_IW6} ${IPP_CORE} ${IPP_CORE_TL_TBB} ${IPP_I_TL_TBB} ${OpenCV_LIBS} ${ITT_NOTIFY})
endif()

add_executable(mean-blur11-avx2 mean-blur11-avx2.cpp)

if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    target_link_libraries(mean-blur11-avx2 ${IPP_CORE} ${IPP_S} ${IPP_I} ${IPP_CV} ${IPP_CORE_TL_TBB} ${IPP_I_TL_TBB} ${IPP_MT_TL_TBB} ${OpenCV_LIBS})
else()
    target_link_libraries(mean-blur11-avx2 ${IPP_IW1} ${IPP_IW2} ${IPP_CV} ${IPP_I} ${IPP_S} ${IPP_IW6} ${IPP_CORE} ${IPP_CORE_TL_TBB} ${IPP_I_TL_TBB} ${OpenCV_LIBS} ${ITT_NOTIFY})
endif()

add_executable(mean-blur11-sse mean-blur11-sse.cpp)

if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    target_link_libraries(mean-blur11-sse ${IPP_CORE} ${IPP_S} ${IPP_I} ${IPP_CV} ${IPP_CORE_TL_TBB} ${IPP_I_TL_TBB} ${IPP_MT_TL_TBB} ${OpenCV_LIBS})
else()
    target_link_libraries(mean-blur11-sse ${IPP_IW1} ${IPP_IW2} ${IPP_CV} ${IPP_I} ${IPP_S} ${IPP_IW6} ${IPP_CORE} ${IPP_CORE_TL_TBB} ${IPP_I_TL_TBB} ${OpenCV_LIBS} ${ITT_NOTIFY})
endif()

add_executable(mean-blur mean-blur11.cpp)

if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    target_link_libraries(mean-blur ${IPP_CORE} ${IPP_S} ${IPP_I} ${IPP_CV} ${IPP_CORE_TL_TBB} ${IPP_I_TL_TBB} ${IPP_MT_TL_TBB} ${OpenCV_LIBS})
else()
    target_link_libraries(mean-blur ${IPP_IW1} ${IPP_IW2} ${IPP_CV} ${IPP_I} ${IPP_S} ${IPP_IW6} ${IPP_CORE} ${IPP_CORE_TL_TBB} ${IPP_I_TL_TBB} ${OpenCV_LIBS} ${ITT_NOTIFY})
endif()

add_custom_target(run-blur ./mean-blur)