# `Gaussian Blur` Sample

## Introduction
This sample illustrates how CVOI helps to improve the computational efficiency to do the Gaussian Blur. This Gaussian Blur sample code is implemented using C++ for CPU.

## Key Implementation Details
As is mentioned in the description of CVOI, We provide three different levels of optimization methods to help to build high-performance MV solutions. In `Gaussian Blur` case. The Following methods have been utilized:

* **Level One**:  
    Advanced Libraries: [Leverage Intel IPP](#leverage-intel-ipp)
* **Level Two**:  
    Multithreading: [Using oneTBB to implement multithreading](#using-onetbb-to-implement-multithreading)  



### Leverage Intel IPP

To get started with ipp, please refer to [Optimization-Methodologies/Optimized-Libraries/IppGetStarted](../../../../Optimization-Methodologies/Optimized-Libraries/IppGetStarted/README.md)

use ` ippiFilterGaussian_8u_C3R_L ` API to replace the origin CV API

api example refer to [ipp gaussian example ](https://www.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-2-image-processing/filtering-functions-2/fixed-filters/filtergaussian.html)


here is an example code snippet below

```c++
//   A simple example of performing a filtering an image using a general integer rectangular kernel
// implemented with Intel(R) Integrated Primitives (Intel(R) IPP) functions:
//     ippiImageJaehne_32f_C1R
//     ippiFilterGaussianGetSpecSize_L
//     ippiFilterGaussianInit_L
//     ippiFilterGaussianGetBufferSize_L
//     ippiFilterGaussian_32f_C1R_L


#include <stdio.h>
#include "ipp.h"

#define WIDTH  128  /* image width */
#define HEIGHT  64  /* image height */

/* Next two defines are created to simplify code reading and understanding */
#define EXIT_MAIN exitLine:                                  /* Label for Exit */
#define check_sts(st) if((st) != ippStsNoErr) goto exitLine; /* Go to Exit if Intel(R) IPP function returned status different from ippStsNoErr */

/* Results of ippMalloc() are not validated because Intel(R) IPP functions perform bad arguments check and will return an appropriate status  */

int main(void)
{
    IppStatus status = ippStsNoErr;
    Ipp32f* pSrc = NULL, * pDst = NULL;     /* Pointers to source/destination images */
    IppSizeL srcStep = 0, dstStep = 0;      /* Steps, in bytes, through the source/destination images */
    IppiSizeL roiSizeL = { WIDTH, HEIGHT }; /* Size of source/destination ROI in pixels */
    IppiSize roiSize = { WIDTH, HEIGHT };   /* Size of source/destination ROI in pixels - for ImageJaehne only */
    int kernelSize = 3;
    Ipp32f sigma = 0.35f;
    Ipp8u* pBuffer = NULL;                 /* Pointer to the work buffer */
    Ipp8u* pInitBuf = NULL;                /* Pointer to the Init buffer */
    IppFilterGaussianSpec* pSpec = NULL;   /* context structure */
    IppSizeL iTmpBufSize = 0, iSpecSize = 0, iInitBufSize = 0;    /* Common work buffer size */
    IppiBorderType borderType = ippBorderRepl;
    Ipp32f borderValue = 0;
    int numChannels = 1;

    pSrc = ippiMalloc_32f_C1_L(roiSize.width, roiSize.height, &srcStep);
    pDst = ippiMalloc_32f_C1_L(roiSize.width, roiSize.height, &dstStep);

    check_sts(status = ippiImageJaehne_32f_C1R(pSrc, (int)srcStep, roiSize)) /* fill source image */

    check_sts(status = ippiFilterGaussianGetSpecSize_L(kernelSize, ipp32f, numChannels,
        &iSpecSize, &iInitBufSize))

    pSpec = (IppFilterGaussianSpec*)ippsMalloc_8u_L(iSpecSize);
    pInitBuf = ippsMalloc_8u_L(iInitBufSize);

    status = ippiFilterGaussianInit_L(roiSizeL, kernelSize, sigma, borderType,
        ipp32f, numChannels, pSpec, pInitBuf);
    ippsFree(pInitBuf);
    check_sts(status)

    check_sts(status = ippiFilterGaussianGetBufferSize_L(roiSizeL, kernelSize, ipp32f, borderType,
        numChannels, &iTmpBufSize))

    pBuffer = ippsMalloc_8u_L(iTmpBufSize);

    check_sts(status = ippiFilterGaussian_32f_C1R_L(pSrc, srcStep, pDst, dstStep, roiSizeL,
        borderType, &borderValue, pSpec, pBuffer))

    EXIT_MAIN
    ippsFree(pBuffer);
    ippsFree(pSpec);
    ippiFree(pSrc);
    ippiFree(pDst);
    printf("Exit status %d (%s)\n", (int)status, ippGetStatusString(status));
    return (int)status;
}
```


### Using oneTBB to implement multithreading

To get started with using oneTBB, please refer to [Optimization-Methodologies/Optimized-Libraries/IppMultithreading](../../../../Optimization-Methodologies/Optimized-Libraries/IppMultithreading/README.md)

Take advantage of ROI (Region Of Interest) support by ` ippiFilterGaussian_8u_C3R_L `, we use oneTBB to parallelize the gaussian blur.






---

## Building the `Gaussian Blur ` Sample Program  
  

### Requirements


| Item                    | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 20.04; Windows 10
| Hardware                          | 11th Intel Core Processor
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 15 minutes




### On a Linux System
    * Build Gaussian Blur Sample program
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

#### Visual Studio IDE
     * Open Visual Studio 2019
     * Select Menu "File > Open > Project/Solution", find "GaussianBlur" folder and select "GaussianBlur.sln"
     * Select Menu "Project > Build" to build the selected configuration, please make sure use x64 platform
     * Select Menu "Debug > Start Without Debugging" to run the program

#### Command line to generate the project file manually * (Option)
     * Make sure CMake has been installed to your OS
     * Execute CMake(cmake-gui) tool, select the src code and binary folders both to the absolute path of "GaussianBlur", and Click the "Configure" and "Generate"

## Running the Sample

### Application Parameters

None

### Example of Output
```bash
=======  opencv gaussian blur START ===========
opencv gaussian filter took 59 milliseconds

=======  tbb ipp platform aware gaussian START ===========
chunksize is 2160
tbb ipp platform aware gaussian filter took 49 milliseconds
pixels correctness is 99.42 %
```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)
