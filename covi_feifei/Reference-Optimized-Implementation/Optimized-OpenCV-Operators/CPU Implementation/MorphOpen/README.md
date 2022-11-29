# `Morphology-Open` Sample
## Introduction
This sample illustrates how CVOI helps to improve the computational efficiency to execute the Morphology-Open. This Morphology-Open sample code is implemented using C++ for CPU.

## Key Implementation Details
As is mentioned in the description of CVOI, We provide three different levels of optimization methods to help to build high-performance MV solutions. In `Morphology-Open ` case. The Following methods have been utilized:

* **Level One**: 
    1. Advanced Libraries: [Leverage Intel IPP](#leverage-intel-ipp)
* **Level Two**:

    1. Multithreading: [Using oneTBB to implement multithreading](#using-onetbb-to-implement-multithreading)  

### Leverage Intel IPP
To get started with the usage of Intel ipp, please refer to [get started with ipp](../../../../Optimization-Methodologies/Optimized-Libraries/IppGetStarted/).

Here we use `ippiMorphOpen_8u_C1R_L ` API to replace the original CV API. More details refers to [MorphOpen](https://www.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-2-image-processing/morphological-operations/morphopen.html).

```
 status = ippiMorphOpen_8u_C1R_L(pSrc, step8, ipp_dst, step8, roiSize, ippBorderDefault, NULL, m_specOpen, tmpBuffer);
```

### Using oneTBB to implement multithreading
For more multithreading implementaion please refer to [IPP with Multithreading](../../../../Optimization-Methodologies/Optimized-Libraries/IppMultithreading/).

Taking advantage of the ROI (Region Of Interest) support by IPP, here we leverage the oneTBB to implement multithreading along with ipp. We first split the original image into subfigures according to the number of physical cores. Then we execute the Morphology Open operation on these subfigures concurrently.

---
## Building the Morphology-Open Sample Program


### Requirement

| Item                    | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 20.04; Windows 10
| Hardware                          | 11th Intel Core Processor
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 15 minutes

### On a Linux System
    * Build Morphology-Open Sample program
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/ipp/latest/lib/intel64/tl/tbb
      source /opt/intel/oneapi/setvars.sh
      source /opt/intel/openvino_2021/bin/setupvars.sh
      mkdir build &&
      cd build &&
      cmake .. &&
      make VERBOSE=1

    * Run the program
      make run-open-3
      make run-open-5

    * Clean the program
      make clean

### On a Windows System
     * Add below directories to the system environment PATH variable befor running the program
         C:\Program Files (x86)\IntelSWTools\openvino_2021\opencv\bin (<your opencv bin directory>)
         C:\Program Files (x86)\Intel\oneAPI\tbb\2021.1.1\redist\intel64\vc_mt
         C:\Program Files (x86)\Intel\oneAPI\ipp\2021.1.1\redist\intel64
         C:\Program Files (x86)\Intel\oneAPI\ipp\2021.1.1\redist\intel64\tl\tbb

### Using Visual Studio Code*  (Optional)

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

#### Visual Studio IDE
     * Open Visual Studio
     * Select Menu "File > Open > Project/Solution", find "MorphOpen" folder and select "MorphOpen.sln"
     * Check your project properties in C/C++ and Linker part to make sure you add the correct include directory and libraries
     * Select Menu "Project > Build" to build the selected configuration, please make sure use x64 platform
     * Select Menu "Debug > Start Without Debugging" to run the program

#### Command line to generate the project file manually * (Option)
     * Make sure CMake has been installed to your OS
     * Execute CMake(cmake-gui) tool, select the src code and binary folders both to the absolute path of "MorphOpen", and Click the "Configure" and "Generate"

## Running the Sample

### Application Parameters

None

### Example of Output
```bash
/************************Performance evaluation starts************************/
CV Time consuming:0.97958
IPP Time consuming:0.78626
pixels correctness is 100.00 % 
tbb ipp time consuming:0.24445
pixels correctness is 100.00 % 
/************************Performance evaluation ends************************/

```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)
If you have trouble building the project with Visual Studio, check your configuration of your project, including environment variables, include directory and linked dynamic libraries.