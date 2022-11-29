# IPP Get Started Sample
[IPP](https://www.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top.html) 
means "Integrated Performance Primitives", a 
software library that provides highly optimized 
functions for signal, data, and image processing.

[IW IPP](https://www.intel.com/content/www/us/en/develop/documentation/ippiw-dev-guide-and-reference/top/introducing-integration-wrappers-for-intel-integrated-performance-primitives.html#introducing-integration-wrappers-for-intel-integrated-performance-primitives) 
means "Integration Wrappers Integrated 
Performance Primitives", aggregate Intel IPP 
functionality in easy-to-use functions and help 
to reduce effort required to integrate Intel IPP 
into your code.

In this example, we demonstrate how to implement 
"Image Add" function with 3 kinds of APIs, including 
OpenCV, IPP, and IW IPP.

# Dependencies
| Dependent Software | Download Source |
| :---: | :---: |
| IPP | [Intel® OneAPI Base Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/rendering-toolkit-download.html) |
| IW IPP | [Intel® OneAPI Base Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/rendering-toolkit-download.html) |
| OpenCV | [Intel® Distribution of OpenVINO™ Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) |

 - OpenCV is not included in OpenVINO toolkit by 
 default anymore. It should be 
 installed/downloaded separately using download 
 script located in "extras/scripts".

# Documentation
| Operator | Supported Intel Architecture | Source |
| :---: | :---: | :---: |
| cv:add | CPU | [cv:add](https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga10ac1bfb180e2cfda1701d06c24fdbd6) |
| ippiAdd_8u_C1RSfs | CPU | [ippiAdd_8u_C1RSfs](https://www.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-2-image-processing/image-arithmetic-and-logical-operations/arithmetic-operations/add-2.html) |
| ipp:iwiAdd | CPU | [ipp:iwiAdd](https://www.intel.com/content/www/us/en/develop/documentation/ippiw-dev-guide-and-reference/top/c-reference-1/image-processing-1/arithmetic-operations-1/iwiadd-1.html) |

# Environment Requirements
| Category | Item |
| :---: | :---: |
| Hardware | Intel@ Series Platform |
| Operating System | Windows 10 |
| Software | [Intel® OneAPI Base Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/rendering-toolkit-download.html), [Intel® Distribution of OpenVINO™ Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html) |

# How to build and run
Build this project with X64/Release version in Microsoft Visual Studio 2022.
