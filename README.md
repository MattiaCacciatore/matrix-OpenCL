# Matrix OpenCL
Matrix multiplication using OpenCL (open source) library and functions to speed them up with a GPU.

## Usage

<details><summary><b>Show instructions</b></summary>

1. Download and install OpenCL SDK (https://github.com/KhronosGroup/OpenCL-SDK) and check all dependencies such as external (you have to download them as well
and put in the right folder, for example OpenCL-CLHPP will go in OpenCL-SDK-main/external/OpenCL-CLHPP).

2. Check in your file system where is OpenCL.lib and copy the path so you can compile the source code.

```sh
$ \path_to\g++.exe -o matrix_opencl matrix_opencl.cpp -I"C:\OpenCL-SDK-main\external\OpenCL-Headers" -L"\path_to_OpenCL.lib_or_OpenCL.so" -lOpenCL
  
$ matrix_opencl.exe
```
Now you should be fine.

3. Compile in Unix (not tested):
  ```sh
  $ $ g++ matrix_opencl.cpp -o matrix_opencl -I"\usr\include" -L"\path_to_OpenCL.lib_or_OpenCL.so" -lOpenCL
  
  $ ./a.out
  ```
</details>

## Motivation

I like working in multi-thread envirorment, it's hard. I was interested in the 'GPU vs CPU performance' topic using an open source library 
and wanted to see for myself.
