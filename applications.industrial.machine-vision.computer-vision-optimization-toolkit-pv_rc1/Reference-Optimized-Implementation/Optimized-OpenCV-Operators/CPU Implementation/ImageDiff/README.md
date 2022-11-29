# `Image Difference` Sample

## Introduction
This sample illustrates how CVOI helps to improve the computational efficiency to compute _Image Difference_. This _Image Difference_ sample code is implemented using C++ for CPU.

## Key Implementation Details
As is mentioned in the description of CVOI, We provide three different levels of optimization methods to help to build high-performance MV solutions. In `Image Difference` case. The Following methods have been utilized:

* **Level One**:  
    Advanced Libraries: [Leverage Intel IPP](#leverage-intel-ipp)
* **Level Two**:  
    Multithreading: [Using oneTBB to implement multithreading](#using-onetbb-to-implement-multithreading)  



### Leverage Intel IPP

To get started with ipp, please refer to [Optimization-Methodologies/Optimized-Libraries/IppGetStarted](../../../../Optimization-Methodologies/Optimized-Libraries/IppGetStarted/README.md)

use ` ippiAbsDiff_8u_C3R ` API to replace the origin CV API

api example refer to [ipp AbsDiff example ](https://www.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-2-image-processing/image-arithmetic-and-logical-operations/arithmetic-operations/absdiff.html)


here is Syntax below

**Syntax**

IppStatus ippiAbsDiff_<mod>(const Ipp<datatype>* pSrc1, int src1Step, const Ipp<datatype>* pSrc2, int src2Step, Ipp<datatype>* pDst, int dstStep, IppiSize roiSize);

Supported values for mod:

8u_C1R
16u_C1R 
32f_C1R 
8u_C3R

### Using oneTBB to implement multithreading

To get started with using oneTBB, please refer to [Optimization-Methodologies/Optimized-Libraries/IppMultithreading](../../../../Optimization-Methodologies/Optimized-Libraries/IppMultithreading/README.md)


---

## Building the `Image Difference ` Sample Program  
  

### Requirements


| Item                    | Description                                               
|:---                               |:----------------------------------------------------------
| OS                                | Linux* Ubuntu* 20.04; Windows 10                          
| Hardware                          | Xeon(R) CPU E5-2678/11th Gen Intel(R) Core(TM) i7-1185G7E 
| Software                          | Intel&reg; OneAPI 2022.1(IPP 2021.5.0) /OpenCV 4.6.0
| Time to complete                  | 15 minutes                                                






## Build and run on Linux

```
mkdir build
cd build
source <IPP_Installed_Dir>/env/vars.sh
cmake ..
make
```

### Run


```
./absdiff_opt
```


## Build and run on Windows
- Open the Visual Studio Solution
- Add your OpenCV and IPP Include path to Visual Studio `Include Directories`
  - `C:\your_opencv_path\opencv\build\include`
  - `C:\Program Files (x86)\Intel\oneAPI\ipp\latest\include`
- Add your OpenCV and IPP Library path to Visual Studio `Library Directories`
  - `C:\your_opencv_path\opencv\build\x64\vc15\lib`
  - `C:\Program Files (x86)\Intel\oneAPI\ipp\latest\lib\intel64`
- *Optional for Visual Studio Debug* Add your OpenCV and IPP Runtime Library path to Visual Studio `Debugging Environment`
  - `PATH=$PATH;C:\your_opencv_path\opencv\build\x64\vc15\bin;C:\Program Files (x86)\Intel\oneAPI\ipp\latest\redist\intel64`
- Build the Solution
- Run the program


