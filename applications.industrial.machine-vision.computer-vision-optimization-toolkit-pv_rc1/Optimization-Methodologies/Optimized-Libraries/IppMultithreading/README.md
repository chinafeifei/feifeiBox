# IPP Multithreading Sample
Multithreading is a model of program execution that allows for multiple threads to be created within a process, executing independently but concurrently sharing process resources. Depending on the hardware, threads can run fully parallel if they are distributed to their own CPU core. Developers leverage threads for maximum application performance and responsiveness.

Intel IPP provides its own threading layer for its API such as *ippiAdd_8u_C1RSfs_LT*.

This sample code demos how to use the IPP with its threading layer and leverage threads using OpenMP and Intel OneTBB.

# Dependencies
The sample application depends on [Intel® OneAPI Base Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html).

# Documentation
| Operator name                             | Supported Intel(r) Architecture(s) | Description
|:---                                       |:---                                |:---
| ipp image add C++                             |   CPU                              | image add as example to show Multithreading with ipp
# System requirements

**Operating System:**
* Windows 10

**Software:**
* [OneAPI Base Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html)
* Visual Studio 2022

**Hardware:**
* Intel® platforms supported by the OneAPI Toolkits.

# How to build and run
 * Open Visual Studio
 * Select Menu "File > Open > Project/Solution", find "IppMultithreading" folder and select "IPP-multi-thread.sln"
 * Check your project properties in C/C++ and Linker part to make sure you add the correct include directory and libraries
 * Select Menu "Project > Build" to build the selected configuration, please make sure use x64 platform
 * Select Menu "Debug > Start Without Debugging" to run the program


## Running the Sample
Example of output
```
/************************Ipp Single & Multi thread example************************/
This an ipp single thread example
This an ipp single thread example : Done, Great !
This an ipp multi thread example: Using TL layer
This an ipp multi thread example: Using TL layer, Done, Great !
This an ipp multi thread example: Using OpenMP
This an ipp multi thread example: Using OpenMP, Done, Great !
This an ipp multi thread example: Using TBB
This an ipp multi thread example: Using TBB, Done Great !
/************************Ipp Single & Multi thread example************************/
```