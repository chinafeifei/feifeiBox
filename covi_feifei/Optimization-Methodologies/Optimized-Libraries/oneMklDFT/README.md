# oneMKL Get Started with DFT Sample
[oneMKL](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html)
means "oneAPI Math Kernel Library" helps you 
achieve maximum performance with a math 
computing library of highly optimized, 
extensively parallelized routines for CPU and 
GPU. The library has C and Fortran interfaces for
most routines on CPU, and DPC++ interfaces for 
some routines on both CPU and GPU.

[DFT](https://en.wikipedia.org/wiki/Discrete_Fourier_transform)
In mathematics, the discrete Fourier transform
(DFT) converts a finite sequence of
equally-spaced samples of a function into a
same-length sequence of equally-spaced samples of
the discrete-time Fourier transform (DTFT), which
is a complex-valued function of frequency. Click
[DFT](https://en.wikipedia.org/wiki/Discrete_Fourier_transform)
for more detail.

In this example, we need a bmp image as input,
and we demonstrate how to implement forward DFT
and inverse DFT with both oneMKL and OpenCV. You
can see a performace improvement on most of Intel
platforms.

# Dependencies
| Dependent Software | Download Source |
| :---: | :---: |
| oneMKL | [Intel® OneAPI Base Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/rendering-toolkit-download.html) |
| OpenCV | [Intel® Distribution of OpenVINO™ Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) |

 - OpenCV is not included in OpenVINO toolkit by 
 default anymore. It should be 
 installed/downloaded separately using download 
 script located in "extras/scripts".

# Table of contents
  * [License](#license)
  * [Documentation](#documentation)
  * [Environment Requirements](#environment-requirements)
  * [How to build and run](#how-to-build-and-run)

# License
The sample application is licensed under MIT license. See [LICENSE](./video_e2e_sample/LICENSE) for details.

# Documentation
| Operator name | Supported Intel(r) Architecture(s) | Description |
| :---: |:---: |:---: |
| DFT/iDFT C | CPU | The calculations of DFT/iDFT in oneMKL C API. Multi-threading acceleration is supported in DFT/iDFT C API. Choose "Parallel" option to enable Multi-threading acceleration ![image](https://github.com/intel-innersource/applications.industrial.machine-vision.computer-vision-optimization-toolkit/assets/89963992/a21b48c2-be2b-4974-8450-e5804e2b28e2) |
| DFT/iDFT DPCPP | CPU GPU | The calculations of DFT/iDFT in oneMKL DPCPP API. DFT/iDFT DPCPP API is supported  on CPU and GPU device. On GPU device,DFT/iDFT always work in parallel.  ![image](https://github.com/intel-innersource/applications.industrial.machine-vision.computer-vision-optimization-toolkit/assets/89963992/93afe07a-c964-495c-b377-4a9ce80fb457)  Currently, Multi-threading acceleration isn't supported on CPU in Windows OS. Even,choose "Parallel" option,DFT/IDFT still work in single thread mode on CPU device. |

# Environment Requirements
| Category | Item |
| :---: | :---: |
| Hardware | Intel@ Series Platform supported by the OneAPI toolkits |
| Operating System | Windows 10 |
| Software | [Intel® OneAPI Base Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/rendering-toolkit-download.html), [Intel® Distribution of OpenVINO™ Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html), [OpenCV 3.14](https://opencv.org/)|


# How to build and run
Run oneMKL_dft.vcxproj in Microsoft Visual Studio 2019.

Change the Include directory and Library dependency directory in project property dialog "VC++ Directories -> Include Directories" 
and "VC++ Directories -> Library Directories", to change the OpenCV and oneMKL include and library folder path on your system.
Build this project with X64/Release version.

change the Debugging->enviroment option with:
* In dft_c project PATH=XX\opencv\build\x64\vc14\bin;C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\redist\intel64_win\compiler;C:\Program Files (x86)\Intel\oneAPI\mkl\latest\redist\intel64;%PATH%
* In dft_dpcpp project PATH=XX\opencv\build\x64\vc14\bin;
make the system can find the library it wants to link.
then run with "Debug->Start without Debugging" to run the application.
