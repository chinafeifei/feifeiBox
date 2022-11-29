# `PCBA Preprocessing` Sample
This sample illustrates how to use CVOI to improve the preprocessing stage of a PCBA defect detection Project. The Optimized part contains Mean blur, Resize, Adaptive Threshold and Morphology Close. This sample code is implemented using C++ for CPU



| Optimized for                     | Description
|:---                               |:---
| OS                                | Windows 10
| Hardware                          | TigerLake with Iris Xe or newer
| Software                          | Intel&reg; C++ Compiler
| What you will learn               | How to use ipp cv api to speed up program
| Time to complete                  | 15 minutes

## Purpose
The purpose of this sample is to show how to implement optimization method based on CVOI (Operator replacement with Optimized ones) into an actual production project and finally improve the efficiency.

## Key Implementation Details
The implementation based on IPP and TBB.




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

## Building the PCBA Preprocess Program

### On a Windows System
     * Add below directories to the system environment PATH variable before running the program
         C:\Program Files (x86)\IntelSWTools\openvino_2021\opencv\bin
         C:\Program Files (x86)\Intel\oneAPI\ipp\2021.1.1\redist\intel64
         C:\Program Files (x86)\Intel\oneAPI\ipp\2021.1.1\redist\intel64\tl\openmp


#### Visual Studio IDE
     * Open Visual Studio 2019
     * Select Menu "File > Open > Project/Solution", find "pcb_opencv" folder and select "pcb_opencv.sln"
     * Select Menu "Project > Build" to build the selected configuration, please make sure use x64 platform
     * Select Menu "Debug > Start Without Debugging" to run the program



## Running the Sample

### Application Parameters

None

### Example of Output
```bash
PCBA Preprocessing

Average opencv time is 3.515 millisecond

Average ipp    time is 2.587 millisecond

Ipp speeds up 135.9 %
```

### Troubleshooting
If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)
