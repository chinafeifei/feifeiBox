//==============================================================
/* Copyright(C) 2022 Intel Corporation
* Licensed under the Intel Proprietary License
*/
// =============================================================

#include <iostream>
#include <vector>
#include <omp.h>
#include <limits>
#include <chrono>
#include <ittnotify.h>

__itt_domain* domain = NULL;
__itt_string_handle* handle_vector = NULL;

#define VECTOR_SIZE (7680 * 4320)

int main()
{
    // Create a domain that is visible globally: we will use it in our example.
    domain = __itt_domain_createW(L"Example.Domain.Global");
    //if (domain != NULL) {
    //    domain->flags = 0; /*disable domain*/
    //}
    handle_vector = __itt_string_handle_create(L"vector_init");

    int mock_result = 0;
    // timing variables
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double, std::milli> delta_ms;
    long total_time = 0;
    long min_time = LONG_MAX;

    std::vector<int> data;

#define REPEAT (1)

    for (int i = 0; i < REPEAT; ++i)
    {
        __itt_task_begin(domain, __itt_null, __itt_null, handle_vector);
        start = std::chrono::high_resolution_clock::now();

        for (int j = 0; j < VECTOR_SIZE; j++) {
            data.push_back(i);
        }
#pragma omp parallel for
        for (int j = 0; j < VECTOR_SIZE; j++) {
            int tid = omp_get_thread_num();
            data[i] += tid;
        }
        end = std::chrono::high_resolution_clock::now();
        __itt_task_end(domain);

        delta_ms = end - start;
        total_time += delta_ms.count();
        if (min_time > delta_ms.count()) {
            min_time = delta_ms.count();
        }

        mock_result += data[rand() % VECTOR_SIZE];
        data.clear();
    }


    std::cout << "Test repeated " << REPEAT << " times." << std::endl;
    std::cout << "Processing std::vector of " << VECTOR_SIZE / (1000000.0) << " MB, Minmum Time: " << min_time
        << ", Average Time: " << total_time / REPEAT << " ms" << std::endl;
    return mock_result;
}
