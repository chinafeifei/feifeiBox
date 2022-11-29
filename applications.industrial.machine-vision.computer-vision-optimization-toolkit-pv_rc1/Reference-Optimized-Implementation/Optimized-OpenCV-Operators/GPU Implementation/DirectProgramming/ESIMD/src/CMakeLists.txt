if (${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
    include_directories(
        ${PROJECT_SOURCE_DIR}/src/esimd_test_utils.hpp
    )
    set(CMAKE_CXX_FLAGS "-fsyc")
else()
    include_directories(
        ${PROJECT_SOURCE_DIR}/src/esimd_test_utils.hpp
    )
    set(CMAKE_CXX_FLAGS "-fsycl")
endif()

# Set default build type to RelWithDebInfo if not specified
# if (NOT CMAKE_BUILD_TYPE)
#     message (STATUS "Default CMAKE_BUILD_TYPE not set. Using Release with Debug Info")
#     set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
#         STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
#         FORCE)
# endif()

add_executable (histogram histogram.cpp)
target_link_libraries(histogram sycl)
add_custom_target (run-histogram ./histogram)