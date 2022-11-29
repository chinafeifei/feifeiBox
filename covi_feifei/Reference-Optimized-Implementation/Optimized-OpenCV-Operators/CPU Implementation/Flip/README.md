# `Flip` Sample
## Introduction

This sample illustrates how CVOI help to improve the computational efficiency to Flip image . This sample code is implemented using C++ for CPU. The purpose of this sample is to show how to convert Flip image by using IPP, with executing on single thread code.

## Key Implementation Details
The implementation based on IPP tasks.

As is mentioned in the description of CVOI, We provide three different levels of optimization methods to help to build high-performance MV solutions. In `Flip` case. The Following methods have been utilized:

* **Level One**: 
  1. Advanced Libraries: [Leverage Intel IPP](#leverage-intel-ipp)

### Leverage Intel IPP

To get started with the usage of Intel ipp, please refer to [get started with ipp](https://github.com/intel-innersource/applications.industrial.machine-vision.computer-vision-optimization-toolkit/blob/f89ba55332f0df93732ddb5114812162c6f51826/Optimization-Methodologies/Optimized-Libraries/IppGetStarted).

Here we use `ipp::ippiMirror` API to replace the original CV API. More details refers to [Mirror](https://www.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-2-image-processing/image-geometry-transforms/geometric-transform-functions/mirror.html)

```
 ipp::ippiMirror(pSrc, srcStep, pDst, dstStep, roiSize, flip)
```



## Building the Flip Sample Program

### Requirement

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 20.04; Windows 10
| Hardware                          | TigerLake with Iris Xe or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | How to use IPP to Flip image
| Time to complete                  | 15 minutes

### On a Linux System
    * Build program
      source /opt/intel/oneapi/setvars.sh
      source /opt/intel/openvino_2021/bin/setupvars.sh
      mkdir build &&
      cd build &&
      cmake .. &&
      make VERBOSE=1
    
    * Run the program
      make run-Flip
    
    * Clean the program
      make clean

### On a Windows System
     * Add below directories to the system environment PATH variable befor running the program
         C:\Program Files (x86)\IntelSWTools\openvino_2021\opencv\bin
         C:\Program Files (x86)\Intel\oneAPI\ipp\2021.1.1\redist\intel64

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:

 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

#### Visual Studio IDE

     * Open Visual Studio 2019
     * Select Menu "File > Open > Project/Solution", find "Flip" folder and select "Flip.sln"
     * Select Menu "Project > Build" to build the selected configuration, please make sure use x64 platform
     * Select Menu "Debug > Start Without Debugging" to run the program

#### Command line to generate the project file manually * (Option)
     * Make sure CMake has been installed to your OS
     * Execute CMake(cmake-gui) tool, select the src code and binary folders both to the absolute path of "Flip", and Click the "Configure" and "Generate"

## Running the Sample

### Example of Output

IPP    Image Flip took 60 ms

OpenCV Image Flip took 85 ms

### Troubleshooting

If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)