/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cudaWarp.h"
#include "mat33.h"


// gpuPerspectiveWarp
template <typename T>
void gpuPerspectiveWarp(T *input, T *output, int width, int height,
                        sycl::float3 m0, sycl::float3 m1, sycl::float3 m2,
                        sycl::nd_item<3> item_ct1)
{
        const int x =
            item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
        const int y =
            item_ct1.get_local_range().get(1) * item_ct1.get_group(1) +
            item_ct1.get_local_id(1);

        if( x >= width || y >= height )
		return;

        const sycl::float3 vec = sycl::float3(x, y, 1.0f);

        const sycl::float3 vec_out = sycl::float3(
            m0.x() * vec.x() + m0.y() * vec.y() + m0.z() * vec.z(),
            m1.x() * vec.x() + m1.y() * vec.y() + m1.z() * vec.z(),
            m2.x() * vec.x() + m2.y() * vec.y() + m2.z() * vec.z());

        const int u = vec_out.x();
        const int v = vec_out.y();

        T px;

        px.x() = 0; px.y() = 0;
        px.z() = 0; px.w() = 0;

        if( u < width && v < height && u >= 0 && v >= 0 )
		px = input[v * width + u];
		
     //if( x != u && y != v )
	//	printf("(%i, %i) -> (%i, %i)\n", u, v, x, y);

	output[y * width + x] = px;
} 


// setup the transformation for the CUDA kernel
inline static void invertTransform(sycl::float3 cuda_mat[3],
                                   const float transform[3][3],
                                   bool transform_inverted)
{
	// invert the matrix if it isn't already
	if( !transform_inverted )
	{
		float inv[3][3];

		mat33_inverse(inv, transform);

		for( uint32_t i=0; i < 3; i++ )
		{
                        cuda_mat[i].x() = inv[i][0];
                        cuda_mat[i].y() = inv[i][1];
                        cuda_mat[i].z() = inv[i][2];
                }
	}
	else
	{
		for( uint32_t i=0; i < 3; i++ )
		{
                        cuda_mat[i].x() = transform[i][0];
                        cuda_mat[i].y() = transform[i][1];
                        cuda_mat[i].z() = transform[i][2];
                }
	}
}


// cudaWarpPerspective
int cudaWarpPerspective(sycl::uchar4 *input, sycl::uchar4 *output,
                        uint32_t width, uint32_t height,
                        const float transform[3][3],
                        bool transform_inverted) try {
        if( !input || !output )
                return 17;

        if( width == 0 || height == 0 )
                return 1;

        // setup the transform
        sycl::float3 cuda_mat[3];
        invertTransform(cuda_mat, transform, transform_inverted);

	// launch kernel
        const sycl::range<3> blockDim(1, 8, 8);
        const sycl::range<3> gridDim(1, iDivUp(height, blockDim[1]),
                                     iDivUp(width, blockDim[2]));

        /*
	DPCT1049:1: The workgroup size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the workgroup size if
         * needed.
	*/
        dpct::device_ext &dev_ct1 = dpct::get_current_device();
        sycl::queue &q_ct1 = dev_ct1.default_queue();
        q_ct1.submit([&](sycl::handler &cgh) {
                auto cuda_mat_ct4 = cuda_mat[0];
                auto cuda_mat_ct5 = cuda_mat[1];
                auto cuda_mat_ct6 = cuda_mat[2];

                cgh.parallel_for(
                    sycl::nd_range<3>(gridDim * blockDim, blockDim),
                    [=](sycl::nd_item<3> item_ct1) {
                            gpuPerspectiveWarp(input, output, width, height,
                                               cuda_mat_ct4, cuda_mat_ct5,
                                               cuda_mat_ct6, item_ct1);
                    });
        });
        dev_ct1.queues_wait_and_throw();

        /*
	DPCT1010:2: SYCL uses exceptions to report errors and does not
         * use the error codes. The call was replaced with 0. You need to
         * rewrite this code.
	*/
        return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// cudaWarpPerspective
int cudaWarpPerspective(sycl::float4 *input, sycl::float4 *output,
                        uint32_t width, uint32_t height,
                        const float transform[3][3],
                        bool transform_inverted) try {
        if( !input || !output )
                return 17;

        if( width == 0 || height == 0 )
                return 1;

        // setup the transform
        sycl::float3 cuda_mat[3];
        invertTransform(cuda_mat, transform, transform_inverted);

	// launch kernel
        const sycl::range<3> blockDim(1, 8, 8);
        const sycl::range<3> gridDim(1, iDivUp(height, blockDim[1]),
                                     iDivUp(width, blockDim[2]));

        /*
	DPCT1049:3: The workgroup size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the workgroup size if
         * needed.
	*/
        dpct::device_ext &dev_ct1 = dpct::get_current_device();
        sycl::queue &q_ct1 = dev_ct1.default_queue();
        q_ct1.submit([&](sycl::handler &cgh) {
                auto cuda_mat_ct4 = cuda_mat[0];
                auto cuda_mat_ct5 = cuda_mat[1];
                auto cuda_mat_ct6 = cuda_mat[2];

                cgh.parallel_for(
                    sycl::nd_range<3>(gridDim * blockDim, blockDim),
                    [=](sycl::nd_item<3> item_ct1) {
                            gpuPerspectiveWarp(input, output, width, height,
                                               cuda_mat_ct4, cuda_mat_ct5,
                                               cuda_mat_ct6, item_ct1);
                    });
        });

        /*
	DPCT1010:4: SYCL uses exceptions to report errors and does not
         * use the error codes. The call was replaced with 0. You need to
         * rewrite this code.
	*/
        return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// cudaWarpAffine
int cudaWarpAffine(sycl::float4 *input, sycl::float4 *output, uint32_t width,
                   uint32_t height, const float transform[2][3],
                   bool transform_inverted) try {
        float psp_transform[3][3];

	// convert the affine transform to 3x3
	for( uint32_t i=0; i < 2; i++ )
		for( uint32_t j=0; j < 3; j++ )
			psp_transform[i][j] = transform[i][j];

	psp_transform[2][0] = 0;
	psp_transform[2][1] = 0;
	psp_transform[2][2] = 1;

	return cudaWarpPerspective(input, output, width, height, psp_transform, transform_inverted);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// cudaWarpAffine
int cudaWarpAffine(sycl::uchar4 *input, sycl::uchar4 *output, uint32_t width,
                   uint32_t height, const float transform[2][3],
                   bool transform_inverted) try {
        float psp_transform[3][3];

	// convert the affine transform to 3x3
	for( uint32_t i=0; i < 2; i++ )
		for( uint32_t j=0; j < 3; j++ )
			psp_transform[i][j] = transform[i][j];

	psp_transform[2][0] = 0;
	psp_transform[2][1] = 0;
	psp_transform[2][2] = 1;

	return cudaWarpPerspective(input, output, width, height, psp_transform, transform_inverted);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
