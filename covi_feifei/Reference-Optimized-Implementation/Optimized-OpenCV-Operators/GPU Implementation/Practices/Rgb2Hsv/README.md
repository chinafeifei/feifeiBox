# Rgb to Hsv in DPC++
## Introduction
This sample illustrates how CVOI helps to improve the computational efficiency to execute the Rgb to Hsv. This Rgb to Hsv sample code is implemented using C++ for GPU.

## Key Implementation Details
As is mentioned in the description of CVOI, We provide three different levels of optimization methods to help to build high-performance MV solutions. In `Rgb to Hsv` case. The Following methods have been utilized:

* **Level Three**:  
    1. DPC++: [DirectProgramming](#directprogramming)

### DirectProgramming
For more information about DirectProgramming, please refer to [DirectProgramming](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-dpcpp-compiler/top.html).
Here we directly use DPC++ to write the code to implement the Rgb to Hsv. To implement this function, we should first understand the algorithm and then implement it with DPC++.


## Building the Rgb to Hsv Program
### Requirement
This contains the DPC++ code of changing the color of picture from rgb to hsv.

| Implementation on                 | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 20.04
| Hardware                          | Tigerlake with GEN11 / Intel Xe Graphics
| Software                          | Intel® oneAPI DPC++/C++ Compiler 2022.2.0s

---



## Before You Begin

1. There are two ways to source the oneAPI DPC++/C++ Compiler:
    1. Intel® oneAPI DPC++/C++ Compiler is included in the Intel® oneAPI Base Toolkit. If you have not installed the Intel® oneAPI Base Toolkit, follow the instructions in [Install Intel® oneAPI Base Toolkit](../../README.md#install-intel-oneapi-base-toolkit) or in the [Official Installation Guide](https://software.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html).
    1. Intel® oneAPI DPC++/C++ Compiler is also included in latest DPC++ Compiler. If you have not installed latest DPC++ Compiler, follow the instructions in [Install Latest DPC++ Compiler](../../README.md#install-latest-dpc-compiler).

1. If you want to run DPCPP code on Intel® GPU, you should install Intel® GPU Driver. If you have not installed it, follow the instruncions in [Install Intel® GPU Driver](../../README.md#install-intel-gpu-driver).

1. Install Intel® Distribution of OpenVINO™ toolkit (at least 2021.1)  
    See [Setup Intel® Distribution of OpenVINO™ Toolkit](../../README.md#install-intel-distribution-of-openvino-toolkit)

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

## DPC++ Rgb2Hsv
Assuming that you already download or clone a full copy of to which this guide is attached. Follow below directions to compile and run source files.

### On a Linux System
1. Prepare the environment to be able to use the Intel® oneAPI DPC++/C++ Compiler
    - Option 1: Source Intel® oneAPI Base Toolkit
        Set system variables by running __setvars.sh__:
        ```
        $ source /opt/intel/oneapi/setvars.sh
        $ export SYCL_DEVICE_FILTER=opencl:gpu
        ```

    - Option 2: Use latest DPC++/C++ Compiler
        Set system variables by running __startup.sh__:
        ```
        # enter dpcpp_compiler/
        $ cd /tmp/dpcpp_compiler/
        # Active the environment variables:
        $ source startup.sh 
        ```
1. Navigate to this repository `Practices/Rgb2Hsv` subfolder:
     ```
     $ cd <Repo_name>/Practices/Rgb2Hsv
     ```

1. Build the program using the build file.
    ```
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make

    # Clean the program
    $ make clean
   ```
    *Note: If you install the openCV not in default path, or install another veresion of openCV, you should change the path in the [CMakeLists.txt](./src/CMakeLists.txt)*

1. Running the program using the run file.
    ```
    # Run the program
    $ make run-hsv-sycl
    ```

### On a Windows System
     * Add below directories to the system environment PATH variable befor running the program
         C:\Program Files (x86)\IntelSWTools\openvino_2021\opencv\bin (<your opencv bin directory>)
         C:\Program Files (x86)\intel\oneAPI\compiler\2022.0.3\windows\bin

#### Visual Studio IDE
     * Open Visual Studio
     * Select Menu "File > Open > Project/Solution", find "Rgb2Hsv" folder and select "Rgb2Hsv.sln"
     * Check your project properties in VC++ Directories and Linker part to make sure you add the correct include directory and libraries (Opencv/dpct)
     * Select Menu "Project > Build" to build the selected configuration, please make sure use x64 platform
     * Select Menu "Debug > Start Without Debugging" to run the program

#### Command line to generate the project file manually * (Option)
     * Make sure CMake has been installed to your OS
     * Execute CMake(cmake-gui) tool, select the src code and binary folders both to the absolute path of "Rgb2Hsv", and Click the "Configure" and "Generate"


### Example of Output
```bash
    img rows is 25
    img cols is 25
    img step is 75
    total pixel is 625
    
    cvtColor() took 0 milliseconds
    BGR [142,226,214] ==> HSV [ 34, 95,226]
    BGR [140,225,217] ==> HSV [ 33, 96,225]
    BGR [136,222,216] ==> HSV [ 32, 99,222]
    BGR [125,211,203] ==> HSV [ 33,104,211]
    BGR [131,221,208] ==> HSV [ 34,104,221]
        Platform Name: Intel(R) OpenCL HD Graphics
        Platform Version: OpenCL 3.0
        Device Name: Intel(R) Iris(R) Xe Graphics [0x9a49]
        Max Work Group: 512
        Max Compute Units: 80
    
    Pass
```