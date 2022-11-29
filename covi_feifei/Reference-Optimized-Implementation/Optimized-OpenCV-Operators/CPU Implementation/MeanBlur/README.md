# `Mean Blur ` Sample

## Introduction
This sample illustrates how CVOI helps to improve the computational efficiency to execute the 11x11 Mean Blur. This 11x11 Mean Blur sample code is implemented using C++ for CPU.

## Key Implementation Details
As is mentioned in the description of CVOI, We provide three different levels of optimization methods to help to build high-performance MV solutions. In `Mean Blur ` case. The Following methods have been utilized:

* **Level One**: 
    1. Compiler: [Switch to Intel Compiler](#switch-to-intel-compiler) 
    2. Advanced Libraries: [Leverage Intel IPP](#leverage-intel-ipp)
* **Level Two**:
    1. Code Analysis: [Analysis with Vtune Profiler](#analysis-with-vtune-profiler) 
    2. Multithreading: [Using oneTBB to implement multithreading](#using-onetbb-to-implement-multithreading)  
* **Level Three**:
    1. Vectorization(SIMD): [Directly programming with AVX2](#directly-programming-with-avx2)  

### Switch to Intel Compiler 

Here we switch the compiler to intel compiler and add compiler flag `fast`. For more information of the fast flag, please refer to this [link](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/compiler-options/optimization-options/fast.html).

***Syntax***

`-fast`(Linux)

`/fast`(Windows)

### Leverage Intel IPP
To get started with the usage of Intel ipp, please refer to [get started with ipp](../../../../Optimization-Methodologies/Optimized-Libraries/IppGetStarted/).

Here we use `ipp::iwiFilterBox` API to replace the original CV API. More details refers to [FilterBox](https://www.intel.com/content/www/us/en/develop/documentation/ippiw-dev-guide-and-reference/top/c-reference-1/image-processing-1/filtering-functions-1/iwifilterbox-1.html)

```
 ipp::iwiFilterBox(srcImage, cvtImage, ksize);
```

### Analysis with Vtune Profiler
To get more information about how to use vtune to analyze the code please refer to [Vtune Profiler](../../../../Optimization-Methodologies/Using-Compiler-and-Analysis-Tools/VTuneProfiler/)

When we use `ipp::iwiFilterBox` API, we got a performance raise with small kernel e.g. (3x3 ,5x5 and 7x7). However, we got a performance drop when the kernel size increases to 11x11.
Here we use Vtune to look for the reason of performance drop.
We check the Vtune result of assembly code with kernel 3x3 and kernel 11x11. Different register is used. In case of kernel (3x3), AVX2 is used. While only SSE is used with kernel 11x11.


### Using oneTBB to implement multithreading
For more multithreading implementaion please refer to [IPP with Multithreading](../../../../Optimization-Methodologies/Optimized-Libraries/IppMultithreading/)

Taking advantage of the ROI (Region Of Interest) support by IPP, here we leverage the oneTBB to implement multithreading along with ipp. We first split the original image into subfigures according to the number of physical cores. Then we execute the Morphology Open operation on these subfigures concurrently.


### Directly programming with AVX2

First get in touch with AVX2(intrinsics)?  Please refer to [Intrinsics](../../../../Optimization-Methodologies/Vectorization/Intrinsics/).

Different X86 platform has different intrinsics support. 
Here we leverage the AVX2 intrinsics to implement our algorithm. For example to caculate the sum of the rows we use the following code snippet. We packed unsigned 8-bit integers to packed 16-bit integers and store them in a 16-element 256-bit vector and execute the sum.
```
static void x11GetFirstSum_8u_C1R(const Ipp8u *pSrc, Ipp64s srcStep, IppiSize dstRoiSize, Ipp16f *pBuffer)
{
    int h, w;
    const Ipp8u *rPtr1, *rPtr2;
    int width = dstRoiSize.width;
    int xMaskSize = 11;
    int yMaskSize = 11;
    int width16 = ((width + xMaskSize - 1) >> 4) << 4;
    rPtr1 = pSrc;
    Ipp16f *pBufferT = (Ipp16f *)pBuffer;
    for (h = 0; h < width16; h += 16)
    {
        __m256 a;
        __m256i ai = _mm256_setzero_si256();
        __m128i aii;
        rPtr2 = rPtr1;
        for (w = yMaskSize; w--;)
        {
            __m256i mSrc0;
            mSrc0 = _mm256_cvtepu8_epi16(*(__m128i *)(rPtr2 + 0));
            ai = _mm256_add_epi16(ai, mSrc0);
            rPtr2 = (Ipp8u *)((Ipp8u *)rPtr2 + srcStep);
        }
        _mm256_storeu_si256((__m256i *)pBufferT, ai);
        rPtr1 += 16;
        pBufferT += 16;
    }

    for (h = h; h < (width + xMaskSize - 1); h++, rPtr1++)
    {
        rPtr2 = rPtr1;
        *pBufferT = 0.f;
        for (w = yMaskSize; w--;)
        {
            *pBufferT += *rPtr2;
            rPtr2 = (Ipp8u *)((Ipp8u *)rPtr2 + srcStep);
        }
        pBufferT++;
    }
    *pBufferT = 0.f;
}
```

---

## Building the `Mean Blur (XX)` Sample Program

### Requirement

| Item                    | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 20.04; Windows 10
| Hardware                          | 11th Intel Core Processor
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 15 minutes

### On a Linux System
    * Build Meanblur Sample program
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/ipp/latest/lib/intel64/tl/tbb
      source /opt/intel/oneapi/setvars.sh
      source /opt/intel/openvino_2021/bin/setupvars.sh
      mkdir build &&
      cd build &&
      cmake .. &&
      make VERBOSE=1
    
    * Run the program
      make run-blur
    
    * Clean the program
      make clean

### On a Windows System
     * Add below directories to the system environment PATH variable befor running the program
         C:\Program Files (x86)\IntelSWTools\openvino_2021\opencv\bin
         C:\Program Files (x86)\Intel\oneAPI\tbb\2021.1.1\redist\intel64\vc_mt
         C:\Program Files (x86)\Intel\oneAPI\ipp\2021.1.1\redist\intel64
         C:\Program Files (x86)\Intel\oneAPI\ipp\2021.1.1\redist\intel64\tl\tbb
    * Comment the 53th '#pragma comment( lib, __FILE__ "/../../../lib/" _INTEL_PLATFORM "ipp_iw" )' in C:\Program Files (x86)\intel\oneAPI\ipp\latest\include\iw\iw_core.h

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with IntelÂ® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.
#### Visual Studio IDE
     * Open Visual Studio 2019
     * Select Menu "File > Open > Project/Solution", find "MeanBlur" folder and select "MeanBlur.sln"
     * Select Menu "Project > Build" to build the selected configuration, please make sure use x64 platform
     * Select Menu "Debug > Start Without Debugging" to run the program

#### Command line to generate the project file manually * (Option)
     * Make sure CMake has been installed to your OS
     * Execute CMake(cmake-gui) tool, select the src code and binary folders both to the absolute path of "MeanBlur", and Click the "Configure" and "Generate"

## Running the Sample

### Example of Output
```bash
Size 4288 2654

CV Time consuming(kernel size 11): 17.966009

IPP Time consuming(kernel size 11): 27.127390

Own IPP Time consuming(kernel size 11): 14.071280

```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for IntelÂ® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)