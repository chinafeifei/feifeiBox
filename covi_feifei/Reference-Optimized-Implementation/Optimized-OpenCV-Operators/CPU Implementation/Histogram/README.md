# `Histogram` Sample
This sample illustrates how CVOI helps to improve the computational efficiency to do histogram. The purpose of this sample is to show how to do implement histogram by using IPP APIs, with one executing on single thread code and another on multi-thread openMP code.

## Key Implementation Details

As is mentioned in the description of CVOI, We provide three different levels of optimization methods to help to build high-performance MV solutions. In `Histogram` case, the implementation based on IPP and openMP tasks. The Following methods have been utilized:

**Level One**: 

1. Advanced Libraries: [Leverage Intel IPP](#leverage-intel-ipp)

### Leverage Intel IPP

To get started with the usage of Intel ipp, please refer to [get started with ipp](https://github.com/intel-innersource/applications.industrial.machine-vision.computer-vision-optimization-toolkit/blob/f89ba55332f0df93732ddb5114812162c6f51826/Optimization-Methodologies/Optimized-Libraries/IppGetStarted).

Here we use `ipp::ippiHistogram` API to replace the original CV API. More details refers to [Histogram](https://www.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-2-image-processing/image-statistics-functions/histogram.html)

```
 ipp::ippiHistogram( pImg, WIDTH, roi, pHistVec, pHistObj, pBuffer )
```



## Build and Run the Histogram Program 
### Requirement

| Item                    | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 20.04; Windows 10
| Hardware                          | 11th Intel Core Processor
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| Time to complete                  | 15 minutes

### On a Linux System

#### Setup environment

After installed onAPI toolkit, go to `<oneapi_install_path>/oneapi/ipp/latest/`, unzip `components_and_examples_lin.tgz` with command `tar -xzvf components_and_examples_lin.tgz`, then it generates `components` folder.
#### Compile
Copy the source code `ipp_histogram` to `<oneapi_install_path>/oneapi/ipp/latest/components/components/examples_core`.

Copy `Makefile` to `<oneapi_install_path>/oneapi/ipp/latest/components/components/examples_core`.

```
cd  <oneapi_install_path>/oneapi/ipp/latest/components/components/examples_core
sudo su
source <oneapi_install_path>/oneapi/setvars.sh
make
```
It generates application `ipp_histogram` in `<oneapi_install_path>/oneapi/ipp/latest/components/components/examples_core/_build/intel64/release`
#### Run
By default, it tests with a 4000x4000 fake image:
```
./ipp_histogram
```
Or, you can run histogram on a specific resolution fake image, for example 2000x2000:
```
./ipp_histogram -s 2000 2000
```
#### Result
The result are verified on Intel platform TigerLake W-11865MLE, for detail, please refer to https://ark.intel.com/content/www/us/en/ark/products/217368/intel-xeon-w11865mle-processor-24m-cache-up-to-4-50-ghz.html
```
Intel(R) IPP:
  ippCore 2021.5 (r0xb404b011) Nov 11 2021
  ippSP AVX-512F/CD/BW/DQ/VL (k0) 2021.5 (r0xb404b011) Nov 11 2021
  ippIP AVX-512F/CD/BW/DQ/VL (k0) 2021.5 (r0xb404b011) Nov 11 2021
generated image of res 4000 * 4000

histogram_ipp, 4 levels,
0.00    64.00   128.00  192.00  256.00
omp thread 32
histogramtile ipp takes 4111 us ======
hist ipp result:
3984323 4018393 4013167 3984117

histogram openCV takes 12991 us ======
histogram_openCV: 4 levels,
[3984323;
 4018393;
 4013167;
 3984117]

```



### On a Windows System

#### Setup environment

    After installed oneAPI toolkit, go to <oneapi_install_path>/oneapi/ipp/latest/components, 
    unzip components_and_examples_win.zip, then it generates three folders in components folder.

#### Visual Studio IDE

     * Open Visual Studio
     
     * Add below directories to the Additional Include Directories by 
       Property Pages -> C/C++ -> General -> Additional Include Directories.
     1.C:\Program Files (x86)\Intel\oneAPI\ipp\2021.6.0\include
     2.C:\Program Files (x86)\Intel\oneAPI\ipp\2021.6.0\components\common\include
     
     * Open MP Support by Property Pages -> C/C++ -> Language -> Open MP Support -> select Yes(/openmp).
     
     * Select Menu "File > Open > Project/Solution", find "Histogram_Win" folder and select "Histogram.sln"
     
     * Select Menu "Project > Build" to build the selected configuration, please make sure use x64 platform
     
     * Select Menu "Debug > Start Without Debugging" to run the program

#### Result

The result are verified on Intel platform Intel(R) Core(TM) i5-8259U, for detail, please refer to https://ark.intel.com/content/www/us/en/ark/products/135935/intel-core-i58259u-processor-6m-cache-up-to-3-80-ghz.html.

```
generated image of res 4000 * 4000

histogram_ipp, 4 levels,
0.00    64.00   128.00  192.00  256.00
omp thread 16
histogramtile ipp takes 5342 us ======
hist ipp result:
3984323 4018393 4013167 3984117

histogram openCV takes 11882 us ======
histogram_openCV: 4 levels,
[3984323;
 4018393;
 4013167;
 3984117]
```



### Troubleshooting

If an error occurs, troubleshoot the problem using the Diagnostics Utility for Intel® oneAPI Toolkits.
[Learn more](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html)
