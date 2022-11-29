# CV IPP DFT Sample
This is a demo to use Intel IPP library's DFT to replace OpenCV DFT, in most intel platform, you can see a performace improvement.

## Typical workloads
This sample sample demoed how to use Intel IPP library implement forward DFT and inverse DFT, this demo need a bmp image as input, it will do DFT transform on this image, and this program will show the time cost of this transform.

# Dependencies
The sample application depends on [Intel® OneAPI Base Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html), and [OpenCV](https://opencv.org/)

# Table of contents

  * [License](#license)
  * [Documentation](#documentation)
  * [System requirements](#system-requirements)
  * [How to build](#how-to-build)

# License
The sample application is licensed under MIT license. See [LICENSE](https://github.com/intel-innersource/applications.industrial.machine-vision.computer-vision-optimization-toolkit/blob/master/License.txt) for details.

# Documentation
For OneAPI IPP library installation, See [enviroment setup guide](https://github.com/intel-innersource/applications.industrial.machine-vision.computer-vision-optimization-toolkit/tree/master/Setup)

# System requirements

**Operating System:**
* Windows 10

**Software:**
* [OpenCV 3.14](https://opencv.org/)
* [OneAPI Base Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html)

**Hardware:**
* Intel® platforms supported by the OneAPI Toolkits.

# How to build and run

* Run ipp_dft.vcxproj in Microsoft Visual Studio 2019.

* Change the Include directory and Library dependency directory in project property dialog "VC++ Directories -> Include Directories" and "VC++ Directories -> Library Directories", to change the OpenCV and IPP include and library folder path on your system.
* Build this project with X64/Release version.
* Prepare an image file named src.bmp. put it in this project root folder.

* change the Debugging->enviroment option with:
   PATH=C:\Program Files (x86)\Intel\oneAPI\ipp\2021.1.1\redist\intel64;C:\Users\iotg\Downloads\opencv314\x64\vc15\bin;C:\Users\zhaoye\Downloads\opencv-3.4.13\opencv-3.4.13\build1\install\x64\vc15\bin;$(VCRedistPaths)%PATH%
* Make the system can find the library it wants to link.
* Then run with "Debug->Start without Debugging" to run the application.
