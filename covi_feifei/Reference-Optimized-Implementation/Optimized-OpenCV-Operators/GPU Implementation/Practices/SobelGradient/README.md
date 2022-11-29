# DPC++ SobelGradient

## Introduction
This sample illustrates how CVOI helps to improve the computational efficiency to execute the SobelGradient. This SobelGradient sample code is implemented using C++ for GPU.

## Key Implementation Details
As is mentioned in the description of CVOI, We provide three different levels of optimization methods to help to build high-performance MV solutions. In `SobelGradient` case. The Following methods have been utilized:

* **Level Two**:  
    DPCT: [DPCT migration](#dpct-migration)
* **Level Three**:  
    DPC++: [DirectProgramming](#directprogramming)

### DPCT migration
The Intel DPC++ Compatibility Tool (dpct) can migrate projects that include multiple source and header files. This sample contains the result of SobelGradient from CUDA-sample to DPC++ sample code on **Linux** and steps to run the DPCPP SobelGradient. For more details, please refer to [DPCT](https://www.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top.html).
### DirectProgramming
For more information about DirectProgramming, please refer to [DirectProgramming](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-dpcpp-compiler/top.html).
Here when finishing the migration from CUDA code, we need to do some modification on the migrated code to make the program run correctly.



## Building the SobelGradient sample
### Requirement

| Implementation on   | Description                                        |
| :------------------ | :------------------------------------------------- |
| OS                  | Linux* Ubuntu* 20.04;                              |
| Hardware            | Tigerlake with GEN11 / Intel Xe Graphics           |
| Software            | Intel® oneAPI DPC++/C++ Compiler 2022.2.0          |

*Note: Ubuntu 20.04 has better intel gpu support than 18.04, while 18.04 will have problems such as no screen output*

---

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

## Before You Begin

1. There are two ways to source the oneAPI DPC++/C++ Compiler:
    1. Intel® oneAPI DPC++/C++ Compiler is included in the Intel® oneAPI Base Toolkit. If you have not installed the Intel® oneAPI Base Toolkit, follow the instructions in [Install Intel® oneAPI Base Toolkit](../../README.md#install-intel-oneapi-base-toolkit) or in the [Official Installation Guide](https://software.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html).
    1. Intel® oneAPI DPC++/C++ Compiler is also included in latest DPC++ Compiler. If you have not installed latest DPC++ Compiler, follow the instructions in [Install Latest DPC++ Compiler](../../README.md#install-latest-dpc-compiler).

1. If you want to run DPCPP code on Intel® GPU, you should install Intel® GPU Driver. If you have not installed it, follow the instruncions in [Install Intel® GPU Driver](../..//README.md#install-intel-gpu-driver).

1. Install Intel® Distribution of OpenVINO™ toolkit (at least 2021.1)  
    See [Setup Intel® Distribution of OpenVINO™ Toolkit](../../README.md#install-intel-distribution-of-openvino-toolkit)

## DPC++ SobelGradient
Assuming that you already download or clone a full copy of the repository to which this guide is attached. Follow below directions to compile and run some source files.

1. Prepare the environment to be able to use the Intel® oneAPI DPC++/C++ Compiler and Navigate to this repository `Practices/SobelGradient` subfolder:

   ```
   $ source /opt/intel/oneapi/setvars.sh
   $ export SYCL_DEVICE_FILTER=opencl:gpu
   $ cd <Repo_name>/Practices/SobelGradient/
   ```

1. Build and run DPCPP SobelGradient source code  
   Directives to build and run DPCPP Erosion source code migrated from CUDA sample.
   
   ```
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make

    # Run the program
    $ make run-sum
    $ make run-hypot

    # Clean the program
    $ make clean
   ```
   *Note: If you install the openCV not in default path, or install another veresion of openCV, you should change the path in the [CMakeLists.txt](./src/CMakeLists.txt)*

1. Example of Output.
```bash
The sobel sum elapsed time in opencv was 1.15163 ms
The sobel sum elapsed time in gpu was 0.096076 ms
Great!!
```

```bash
The sobel hypot elapsed time in opencv was 1.05377 ms
The sobel hypot elapsed time in gpu was 0.0973772 ms
Great!!
```
