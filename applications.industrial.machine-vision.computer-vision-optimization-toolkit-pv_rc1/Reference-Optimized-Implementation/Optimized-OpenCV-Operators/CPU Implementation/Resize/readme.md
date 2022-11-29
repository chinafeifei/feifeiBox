# `Resize` Sample

## Introduction

This sample illustrates how IPP help improve the computational efficiency to resize image. This Resize sample code is implemented using C++ for CPU.

## Key Implementation Details

The implementation based on IPP tasks.
As is mentioned in the description of CVOI, We provide three different levels of optimization methods to help to build high-performance MV solutions. In `Resize` case. The Following methods have been utilized:

* **Level One**: 
  1. Compiler: [Switch to Intel Compiler](#switch-to-intel-compiler) 
  2. Advanced Libraries: [Leverage Intel IPP](#leverage-intel-ipp)
* **Level Two**:
  1. Multithreading: [Using ThreadLayer to implement multithreading](#using-threadlayer-to-implement-multithreading) 

#### Switch to Intel Compiler 

Here we switch the compiler to intel compiler and add compiler flag `fast`. For more information of the fast flag, please refer to this [link](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/compiler-options/optimization-options/fast.html).

***Syntax***

`-fast`(Linux)

`/fast`(Windows)

#### Leverage Intel IPP

To get started with the usage of Intel ipp, please refer to [get started with ipp](../../../../Optimization-Methodologies/Optimized-Libraries/IppGetStarted/).

Here we use `ippiResizeNearest_8u_C3R` API to replace the original CV API.

#### Using ThreadLayer to implement multithreading

For more multithreading implementaion please refer to [IPP with Multithreading](https://github.com/intel-innersource/applications.industrial.machine-vision.computer-vision-optimization-toolkit/blob/pv_rc1/Optimization-Methodologies/Optimized-Libraries/IppMultithreading).

Here we use ` ippiResizeLinear_8u_C1R_LT` API to replace the original CV API.

## Building the Resize Sample Program

### Requirement

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 20.04; Windows 10
| Hardware                          | KabyLake or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | How to use IPP to resize image
| Time to complete                  | 15 minutes

### Using Visual Studio Code  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:

 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### On a Windows System

     * Add below directories to the system environment PATH variable befor running the program
         C:\Program Files (x86)\IntelSWTools\openvino_2021\opencv\bin
         C:\Program Files (x86)\Intel\oneAPI\ipp\2021.1.1\redist\intel64


#### Visual Studio IDE

     * Open Visual Studio 2019
     * Select Menu "File > Open > Project/Solution", find "Resize" folder and select "Resize.sln"
     * Select Menu "Project > Build" to build the selected configuration, please make sure use x64 platform
     * Select Menu "Debug > Start Without Debugging" to run the program

## Running the Sample

### Example of Output

```
Running 100 times and taking average

IPP  Resize took  0.48ms

OpenCV Resize took 0.39 ms

IPP TL Resize took 0.25ms
```

### Troubleshooting

If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)