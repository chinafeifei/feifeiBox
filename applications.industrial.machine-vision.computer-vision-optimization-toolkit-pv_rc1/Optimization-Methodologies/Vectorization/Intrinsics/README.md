# BKM(Best Known Method) for CV Operator Optimization by using Intel(r) Intrinsic (AVX/AVX2, AVX512)

Single-instruction, multiple-data (SIMD) technologies enable the development of advanced multimedia, signal processing, and modeling applications.

Intel® Advanced Vector Extension (Intel® AVX), is a major enhancement to Intel Architecture. It extends the functionality of previous generations of 128-bit SSE vector instructions and increased the vector register width to support 256-bit operations. The Intel AVX ISA enhancement is focused on float-point instructions. Some 256-bit integer vectors are supported via floating-point to integer and integer to floating-point conversions.

Intel® Advanced Vector Extensions 512 (Intel® AVX-512) are the following set of 512-bit instruction set extensions supported by recent microarchitectures, beginning with Skylake server microarchitecture, and the Intel® Xeon Phi™ processors based on Knights Landing microarchitecture.

-  Intel® AVX-512 Foundation (F)
     -  512-bit vector width.
     -  32 512-bit long vector registers.
     -  Data expand and data compress instructions.
     -  Ternary logic instruction.
     -  8 new 64-bit long mask registers.
     -  Two source cross-lane permute instructions.
     -  Scatter instructions.
     -  Embedded broadcast/rounding.
     -  Transcendental support.

- Intel® AVX-512 Conflict Detection Instructions (CD)
- Intel® AVX-512 Exponential and Reciprocal Instructions (ER)
- Intel® AVX-512 Prefetch Instructions (PF)
- Intel® AVX-512 Byte and Word Instructions (BW)
- Intel® AVX-512 Double Word and Quad Word Instructions (DQ)
    -  New QWORD and Compute and Convert Instructions.

- Intel® AVX-512 Vector Length Extensions (VL)

# Dependencies
Please confirm that the CPU supports the instruction set before running


# Table of contents

  * [License](#license)
  * [Documentation](#documentation)
  * [System requirements](#system-requirements)
  * [How to build](#how-to-build)

# License
The sample application is licensed under MIT license. See [LICENSE](https://github.com/intel-innersource/applications.industrial.machine-vision.computer-vision-optimization-toolkit/blob/master/License.txt) for details.

# Documentation
For more information about Intel intrinsic, See [Intel® 64 and IA-32 Architectures Optimization Reference Manual](http://www.intel.com/content/www/us/en/architecture-and-technology/64-ia-32-architectures-optimization-manual.html)

# System requirements

**Operating System:**
* Windows 10
* Ubuntu 20.04

**Software:**
* Visual Studio 2019
* GCC10.3

**Hardware:**
* Intel® platforms support AVX/AVX2/AVX512 intrinsics

# How to build and run

## For Windows OS
### Case 1: Scalar Average
* Build and Run AVX_AVG.vcxproj in Microsoft Visual Studio 2019.
* Then run with "Debug->Start without Debugging" to run the application.

### Case 2: Scalar Multiply
* Build and Run AVX_Multiply.vcxproj in Microsoft Visual Studio 2019.
* Then run with "Debug->Start without Debugging" to run the application.

### Case 3: Scalar Mandelbrot
* Build and Run AVX_Mandelbrot.vcxproj in Microsoft Visual Studio 2019.
* Then run with "Debug->Start without Debugging" to run the application.

## For Linux OS
### Build and Run
```
mkdir build
cd build
cmake ..
make
```
### Case 1: Scalar Average
```
./avx_avg
```
* Example of output
```
Scalar average: 4.99364e+08
==================Original time (ms): 744.345
AVX2 average: 4.99364e+08
==================AVX2 accelerated (ms): 88.516
AVX512 average: 4.99364e+08
==================AVX512 accelerated (ms): 42.101
```
### Case 2: Scalar Multiply
```
./avx_multiply
```
* Example of output
```
Scalar multiply:
result[0]: 1.03332e+11
result[1]: 1.02938e+11
result[2]: 1.02077e+11
result[3]: 1.03036e+11
result[4]: 1.00711e+11
result[5]: 1.05682e+11
result[6]: 1.0269e+11
result[7]: 1.02364e+11
result[8]: 1.02406e+11
result[9]: 1.05008e+11
result[10]: 1.00757e+11
result[11]: 1.03292e+11
result[12]: 1.03552e+11
result[13]: 1.03866e+11
result[14]: 1.03082e+11
result[15]: 1.04662e+11
==================Original time: 61.023
avx2_result[0]: 1.03332e+11
avx2_result[1]: 1.02937e+11
avx2_result[2]: 1.02077e+11
avx2_result[3]: 1.03035e+11
avx2_result[4]: 1.00711e+11
avx2_result[5]: 1.05682e+11
avx2_result[6]: 1.0269e+11
avx2_result[7]: 1.02364e+11
avx2_result[8]: 1.02407e+11
avx2_result[9]: 1.05008e+11
avx2_result[10]: 1.00758e+11
avx2_result[11]: 1.03292e+11
avx2_result[12]: 1.03552e+11
avx2_result[13]: 1.03866e+11
avx2_result[14]: 1.03082e+11
avx2_result[15]: 1.04662e+11
==================AVX2 accelerated: 7.003
avx512_result[0]: 1.03332e+11
avx512_result[1]: 1.02937e+11
avx512_result[2]: 1.02077e+11
avx512_result[3]: 1.03035e+11
avx512_result[4]: 1.00711e+11
avx512_result[5]: 1.05682e+11
avx512_result[6]: 1.0269e+11
avx512_result[7]: 1.02364e+11
avx512_result[8]: 1.02406e+11
avx512_result[9]: 1.05008e+11
avx512_result[10]: 1.00758e+11
avx512_result[11]: 1.03292e+11
avx512_result[12]: 1.03552e+11
avx512_result[13]: 1.03866e+11
avx512_result[14]: 1.03082e+11
avx512_result[15]: 1.04662e+11
==================AVX512 accelerated: 5.018
```
### Case 3: Scalar Mandelbrot
```
./avx_mandelbrot
```
* Example of output
```
Scalar/AVX2 = 7.130934
AVX2/AVX512 = 1.699475
```
