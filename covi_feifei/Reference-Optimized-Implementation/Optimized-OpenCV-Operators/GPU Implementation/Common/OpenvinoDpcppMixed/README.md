## [PointPillars Sample](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/End-to-end-Workloads/LidarObjectDetection-PointPillars)

---
### Descriptions
This contains all the enssential instructions to locally run PointPillars Sample cloned from https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/End-to-end-Workloads/LidarObjectDetection-PointPillars.

| Test on                           | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 20.04
| Hardware                          | Tigerlake with GEN11 / Intel Xe Graphics
| Software                          | Intel® oneAPI DPC++/C++ Compiler 2021.4.0, Intel® Distribution of OpenVINO™ toolkit 

More details about Intel® oneAPI AI Analytics Toolkit (AI Kit)
 and its internal samples such as PointPillars Sample, see [Intel® oneAPI AI Analytics Toolkit (AI Kit)](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics#intel-oneapi-ai-analytics-toolkit-ai-kit).

---
### Requirements (Local)

To build and run the PointPillars sample, the following libraries have to be installed:

1. Intel® Distribution of OpenVINO™ toolkit (at least 2021.1)  
   See [Setup Intel® Distribution of OpenVINO™ Toolkit](../../README.md#install-intel-distribution-of-openvino-toolkit)
   
2. Intel® oneAPI Base Toolkit (at least 2021.2)  
   See [Setup Intel® oneAPI Base Toolkit](../../README.md#install-intel-oneapi-base-toolkit)
   
3. Boost (including boost::program_options and boost::filesystem library).  
   For Ubuntu, you may install the libboost-all-dev package.

   ```
   $ sudo apt install libboost-all-dev
   ```
4. Optional: If the sample should be run on an Intel GPU, it might be necessary to upgrade the corresponding drivers. Therefore, please consult the following  page: https://github.com/intel/compute-runtime/releases/

---
### Build process (Local):
  Assuming that you already download or clone a full copy of to which this guide is attached. Follow below directions to compile and run some source files forked from subdirectory of [Intel® oneAPI AI Analytics Toolkit (AI Kit)](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics#intel-oneapi-ai-analytics-toolkit-ai-kit).
  
  1. Navigate to [src](./src) folder in this repository `OpenvinoDpcppMixed` subfolder:
     ```
     $ cd <Repo_name>/OpenvinoDpcppMixed/src
     ```

  2. Prepare the environment to be able to use the Intel® Distribution of OpenVINO™ toolkit and oneAPI

     ```
     $ source /opt/intel/openvino_2021/bin/setupvars.sh
     [setupvars.sh] OpenVINO environment initialized 

     $ source /opt/intel/oneapi/setvars.sh
     ```
     
  3. Build the program using the following cmake commands.

     ```
     $ mkdir build && cd build
     $ cmake ../
     $ make
     ```

---
### Running the PointPillars Sample Program

- For single-threaded execution on the host system, please use:

  ```
  $ ./example.exe
  Using Host device (single-threaded CPU)
  Starting PointPillars
  PreProcessing - 88ms
  AnchorMask - 1ms
  PFE Inference - 256ms
  Scattering - 12ms
  RPN Inference - 523ms
  Postprocessing - 3ms
  Done
  Execution time: 887ms
  1 cars detected
  Car: Probability = 0.622535 Position = (24.8561, 12.5615, -0.00769353) Length = 2.42855 Width = 3.61394
  ```

- And to use an Intel® DG1 or integrated graphics, please use:

  ```
  $ ./example.exe --gpu
  Using Intel(R) Iris(R) Xe Graphics [0x9a49]
  Starting PointPillars
  PreProcessing - 139ms
  AnchorMask - 74ms
  PFE Inference - 73ms
  Scattering - 58ms
  RPN Inference - 155ms
  Postprocessing - 411ms
  Done
  Execution time: 913ms
  1 cars detected
  Car: Probability = 0.621854 Position = (24.8566, 12.5615, -0.00754929) Length = 2.42863 Width = 3.61413
  ```

- For using multi-threaded CPU execution, please use:

  ```
  $ ./example.exe --cpu
  Using 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
  Starting PointPillars
  PreProcessing - 156ms
  AnchorMask - 105ms
  PFE Inference - 97ms
  Scattering - 15ms
  RPN Inference - 175ms
  Postprocessing - 405ms
  Done
  Execution time: 956ms
  1 cars detected
  Car: Probability = 0.621526 Position = (24.857, 12.5615, -0.00759459) Length = 2.42868 Width = 3.61427
  ```

- These options can also be used in combination, e.g.:

  ```
  $ ./example.exe --cpu --gpu --host
  Using 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
  Starting PointPillars
  PreProcessing - 106ms
  AnchorMask - 103ms
  PFE Inference - 101ms
  Scattering - 16ms
  RPN Inference - 198ms
  Postprocessing - 415ms
  Done
  Execution time: 943ms
  1 cars detected
  Car: Probability = 0.621555 Position = (24.8569, 12.5615, -0.00759792) Length = 2.42868 Width = 3.61425
  Using Intel(R) Iris(R) Xe Graphics [0x9a49]
  Starting PointPillars
  PreProcessing - 153ms
  AnchorMask - 72ms
  PFE Inference - 80ms
  Scattering - 59ms
  RPN Inference - 163ms
  Postprocessing - 413ms
  Done
  Execution time: 944ms
  1 cars detected
  Car: Probability = 0.62186 Position = (24.8565, 12.5615, -0.00753474) Length = 2.42862 Width = 3.6141
  Using Host device (single-threaded CPU)
  Starting PointPillars
  PreProcessing - 77ms
  AnchorMask - 1ms
  PFE Inference - 254ms
  Scattering - 12ms
  RPN Inference - 515ms
  Postprocessing - 1ms
  Done
  Execution time: 863ms
  1 cars detected
  Car: Probability = 0.622535 Position = (24.8561, 12.5615, -0.00769353) Length = 2.42855 Width = 3.61394
  ```
---
