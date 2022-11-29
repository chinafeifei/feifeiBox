//==============================================================
/* Copyright(C) 2022 Intel Corporation
* Licensed under the Intel Proprietary License
*/
// =============================================================

#include <iostream>
#include <limits>
#include <chrono>
#include <ipps.h>
#include <ittnotify.h>

__itt_domain* domain = NULL;
__itt_string_handle* handle_memcpy = NULL;

#define WIDTH 7680
#define HEIGHT 4320
#define IMAGE_SIZE (WIDTH * HEIGHT * 4)

int main()
{
    // Create a domain that is visible globally: we will use it in our example.
    domain = __itt_domain_createW(L"Example.Domain.Global");
    //if (domain != NULL) {
    //    domain->flags = 0; /*disable domain*/
    //}
    // Create string handles which associates with the "memcpy" task.
    handle_memcpy = __itt_string_handle_create(L"memcpy");

    int mock_result = 0;
    // timing variables
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double, std::milli> delta_ms;
    long total_time = 0;
    long min_time = LONG_MAX;

    char* pImage = NULL;
    char* pBuffer = NULL;

    pImage = (char*)malloc(IMAGE_SIZE);
    pBuffer = (char*)malloc(IMAGE_SIZE);

    if (pImage == NULL || pBuffer == NULL) {
        std::cout << "malloc failed!" << std::endl;
        return -1;
    }

#define REPEAT (3)

    for (int i = 0; i < REPEAT; ++i)
    {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            pImage[j] = rand();
        }

        __itt_task_begin(domain, __itt_null, __itt_null, handle_memcpy);
        start = std::chrono::high_resolution_clock::now();

        // Activity to be timed
        memcpy(pBuffer, pImage, IMAGE_SIZE);

        end = std::chrono::high_resolution_clock::now();
        __itt_task_end(domain);

        delta_ms = end - start;
        total_time += delta_ms.count();
        if (min_time > delta_ms.count()) {
            min_time = delta_ms.count();
        }
        mock_result += pBuffer[rand() % IMAGE_SIZE]; // prevent compiler from optimzing out 'memcpy'
    }

    free(pImage);
    free(pBuffer);
    pImage = NULL;
    pBuffer = NULL;
    std::cout << "memcpy() repeated " << REPEAT << " times." << std::endl;
    std::cout << "Copy image of " << IMAGE_SIZE / (1000000.0) << " MB, Minmum Time: " << min_time
        << ", Average Time: " << total_time / REPEAT << " ms" << std::endl;
    std::cout << "Maximum Memory Bandwidth realized: " << 2 * IMAGE_SIZE / (1000000.0) / (min_time / 1000.0) << "MB/s" << std::endl;

    // reset timer
    total_time = 0;
    min_time = LONG_MAX;

    pImage = (char*)ippsMalloc_64s(IMAGE_SIZE / 8);
    pBuffer = (char*)ippsMalloc_64s(IMAGE_SIZE / 8);

    if (pImage == NULL || pBuffer == NULL) {
        std::cout << "malloc failed!" << std::endl;
        return -1;
    }

    for (int i = 0; i < REPEAT; ++i)
    {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            pImage[j] = rand();
        }

        __itt_task_begin(domain, __itt_null, __itt_null, handle_memcpy);
        start = std::chrono::high_resolution_clock::now();

        // Activity to be timed
        ippsCopy_64s((Ipp64s*)pImage, (Ipp64s*)pBuffer, IMAGE_SIZE / 8);

        end = std::chrono::high_resolution_clock::now();
        __itt_task_end(domain);

        delta_ms = end - start;
        total_time += delta_ms.count();
        if (min_time > delta_ms.count()) {
            min_time = delta_ms.count();
        }
        mock_result += pBuffer[rand() % IMAGE_SIZE]; // prevent compiler from optimzing out 'memcpy'
    }

    ippsFree(pImage);
    ippsFree(pBuffer);
    pImage = NULL;
    pBuffer = NULL;
    std::cout << "ippsCopy() repeated " << REPEAT << " times." << std::endl;
    std::cout << "Copy image of " << IMAGE_SIZE / (1000000.0) << " MB, Minmum Time: " << min_time
        << ", Average Time: " << total_time / REPEAT << " ms" << std::endl;
    std::cout << "Maximum Memory Bandwidth realized: " << 2 * IMAGE_SIZE / (1000000.0) / (min_time / 1000.0) << "MB/s" << std::endl;

    return mock_result;
}
