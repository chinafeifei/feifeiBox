# **Integrate IPP with minimal binary size increase**

## **Introduction**

This tutorial shows how to Integrate IPP with minimal binary size increase.

Intel® Integrated Performance Primitives (Intel® IPP) is distributed as:

- **Static library:** Static linking results in a standalone executable.
- **Dynamic library:** Dynamic linking defers function resolution until run-time and requires that you bundle the redistributable libraries with your application.

## Dependencies

| Dependent Software | Download Source                                              |
| ------------------ | ------------------------------------------------------------ |
| IPP                | [Intel® OneAPI Base Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/rendering-toolkit-download.html) |

## Environment Requirements

| Category         | Item                                                         |
| ---------------- | ------------------------------------------------------------ |
| Hardware         | Intel@ Series Platform                                       |
| Operating System | Windows 10                                                   |
| Software         | [Intel® OneAPI Base Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/rendering-toolkit-download.html) |

## How to Integrate IPP

Use Intel IPP by Property Pages - > Intel Libraries for oneAPI  - >  Intel® Integrated Performance Primitives(Intel® IPP) - > Use Intel® IPP - > select Static Library.



*NOTE: If you want to build a custom dynamic library using the Intel IPP Custom Library Tool, please refer to  [Building a Custom DLL with Custom Library Tool](https://www.intel.com/content/www/us/en/develop/documentation/dev-guide-ipp-for-oneapi/top/ipp-custom-library-tool/building-a-custom-dll-with-custom-library-tool.html).

