# `Sobel Gradient Filter` Sample

## Introduction
This sample illustrates how CVOI helps to improve the computational efficiency to do the Sobel Gradient. This Sobel Gradient sample code is implemented using C++ for CPU.

## Key Implementation Details

As is mentioned in the description of CVOI, We provide three different levels of optimization methods to help to build high-performance MV solutions. In `Sobel Gradient` case. The Following methods have been utilized:

* **Level One**:  
    Advanced Libraries: [Leverage Intel IPP](#leverage-intel-ipp)
 




### Leverage Intel IPP

To get started with ipp, please refer to [Optimization-Methodologies/Optimized-Libraries/IppGetStarted](../../../../Optimization-Methodologies/Optimized-Libraries/IppGetStarted/README.md)

use `ippiFilterSobelHorizBorder` and `ippiFilterSobelVertBorder` API to replace the original CV API.

api example refer to [ipp sobel gradient filter example ](https://www.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-2-image-processing/filtering-functions-2/fixed-filters/filtersobelhorizborder.html)


here is an example code snippet below

```c++
IppStatus fix_sobelhoriz_8u16( void ) {
    Ipp8u pSrc[9*8] =
	    {
	        0, 1, 2, 120, 121, 122, 50, 51, 52,
	        1, 2, 3, 121, 122, 123, 52, 52, 53,
	        3, 4, 5, 130, 131, 132, 63, 64, 65,
	        4, 5, 6, 131, 132, 133, 64, 65, 66,
	        5, 6, 7, 132, 133, 134, 65, 66, 67,
            8, 7, 6, 134, 133, 132, 67, 66, 65,
            7, 6, 5, 133, 132, 131, 66, 65, 64,
            6, 5, 4, 132, 131, 130, 65, 64, 63
          };
    Ipp16s  pDst[8*7];
    Ipp8u   *pBuffer;
    IppiSize roiSize = {8, 7};
    IppiBorderType borderType = ippBorderRepl | ippBorderInMemTop | ippBorderInMemRight;
    int    srcStep = 9 * sizeof(Ipp8u);
    int    dstStep = 8 * sizeof(Ipp16s);
    int    bufferSize;
    IppStatus status;
    ippiFilterSobelHorizBorderGetBufferSize(roiSize, ippMskSize3x3, ipp8u, ipp16s, 1, &bufferSize);
    pBuffer = ippsMalloc_8u(bufferSize);
    status = ippiFilterSobelHorizBorder_8u16s_C1R(pSrc + srcStep, srcStep, pDst, dstStep, roiSize, ippMskSize3x3,
             borderType, 0, pBuffer);
    ippsFree(pBuffer);
    return status;
}

```


---

## Building the `Sobel Gradient Filter` Sample Program



### Requirements


| Item                    | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 20.04; Windows 10
| Hardware                          | 11th Intel Core Processor
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 15 minutes



### On a Linux System
    * Build Sobel Sample program
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/ipp/latest/lib/intel64/tl/tbb
      source /opt/intel/oneapi/setvars.sh
      source /opt/intel/openvino_2021/bin/setupvars.sh
      mkdir build &&
      cd build &&
      cmake .. &&
      make VERBOSE=1

    * Run the program
      make run-sum
      make run-hypot

    * Clean the program
      make clean

### On a Windows System
     * Add below directories to the system environment PATH variable befor running the program
         C:\Program Files (x86)\IntelSWTools\openvino_2021\opencv\bin
         C:\Program Files (x86)\Intel\oneAPI\tbb\2021.1.1\redist\intel64\vc_mt
         C:\Program Files (x86)\Intel\oneAPI\ipp\2021.1.1\redist\intel64
         C:\Program Files (x86)\Intel\oneAPI\ipp\2021.1.1\redist\intel64\tl\tbb

#### Visual Studio IDE
     * Open Visual Studio 2019
     * Select Menu "File > Open > Project/Solution", find "SobelGradient" folder and select "SobelGradient.sln"
     * Select Menu "Project > Build" to build the selected configuration, please make sure use x64 platform
     * Select Menu "Debug > Start Without Debugging" to run the program

#### Command line to generate the project file manually * (Option)
     * Make sure CMake has been installed to your OS
     * Execute CMake(cmake-gui) tool, select the src code and binary folders both to the absolute path of "SobelGradient", and Click the "Configure" and "Generate"

## Running the Sample

### Application Parameters

None

### Example of Output
```bash
=============  OPENCV sobel abs sum =====================
opencv Sobel x took 101 milliseconds


=============  SINGLE thread sobel abs sum ==================
ipp Sobel single thread x took 76 milliseconds


=============  MULTI thread sobel abs sum ==================
ipp Sobel multi thread x took 52 milliseconds


================  compare pass  ======================
compare took 14 milliseconds
```

```bash
============= START openCV vertical ================
opencv Sobel hypot took 280 milliseconds

============= SINGLE thread sobel ==================
ipp Sobel single thread hypot took 196 milliseconds

============= MULTI thread sobel ===================
ipp Sobel multi thread hypot took 60 milliseconds

============= compare pass =========================
compare took 129 milliseconds
```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)