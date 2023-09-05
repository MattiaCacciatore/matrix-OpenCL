// =================================================================================================
// Project: Exploring the performance of general matrix multiplication on GPU. (educational purpose)
//
// Compiler flags for Windows:
// \path_to\g++.exe -o matrix_opencl matrix_opencl.cpp -I"C:\OpenCL-SDK-main\external\OpenCL-Headers" -L"\path_to_OpenCL.lib_or_OpenCL.so" -lOpenCL
// Compiler flags for Unix (not tested):
// $ g++ matrix_opencl.cpp -o matrix_opencl -I"\usr\include" -L"\path_to_OpenCL.lib_or_OpenCL.so" -lOpenCL
//
// -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" <--- default path for NVIDIA (Windows 10)
// -I"C:\OpenCL-SDK-main\external\OpenCL-Headers" <--- my path CL/cl.h header
// -L"C:\clGPU\common\intel_ocl_icd\windows\Debug\lib\x64\OpenCL.lib" <-- path for Intel (not tested)
// =================================================================================================
// 300 -> 3.0 OpenCL version, you can change it (for example 120 and 200).
#define CL_TARGET_OPENCL_VERSION 300
//#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <iostream>
#include <cmath>   // abs().
#include <string>
#include <chrono>
#include <CL/cl.h> // You can find the meaning of OpenCL error codes in this header.
// =================================================================================================
// Size of the matrices - K, M, N (squared).
#define SIZE 4096
// Threadblock sizes.
#define TS 32
// =================================================================================================
// FUNCTIONS
// =================================================================================================
// Set the kernel as a string (better to do this in a separate file though).
// The function that will be executed on the GPU using threads to speed up calculation.
const char* kernelstring =
    "__kernel void gpu_matrix_mult(const int M, const int N, const int K, const __global float* A, const __global float* B, __global float* C){"
    "    const int globalRow = get_global_id(0);"
    "    const int globalCol = get_global_id(1);"
    "    float acc = 0;"
    "    for (int i=0; i<N; i++){"
    "        acc += A[globalRow*N + i] * B[i*K + globalCol];"
    "    }"
    "    C[globalRow*K + globalCol] = acc;"
    "}";
// =================================================================================================
void cpu_matrix_mult(const float* const c_a, const float* const c_b, float* c_c, const int m, const int n, const int k){
    float sum = 0;
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < k; ++j){
            sum = 0;
            for (int h = 0; h < n; ++h){
                sum += c_a[i * n + h] * c_b[h * k + j];
            }
            c_c[i * k + j] = sum;
        }
    }
}
// =================================================================================================
// HELPER FUNCTIONS
// =================================================================================================
void print_matrix(const float* const matrix, const int m, const int n){
    std::cout << "\n";
    for (int i = 0; i < m; ++i){
        std::cout << "[ ";
        for (int j = 0; j < n; ++j){
            std::cout << matrix[i*m+j] << " ";
        }
        std::cout << " ]\n";
    }
    std::cout << "\n";
}
// =================================================================================================
// MAIN
// =================================================================================================
int main(){
//------------------------------------ Initialization ----------------------------------------------
	// Timer.
	std::chrono::high_resolution_clock::time_point start, end;
    	// Set the sizes.
    	const int K = SIZE, M = SIZE, N = SIZE;

    	const size_t local[2] = {TS, TS};
    	const size_t global[2] = {static_cast<size_t>(M), static_cast<size_t>(N)};

	char* messages = NULL;
	size_t log_size = 0;
	double cpu_elapsed_time_ms = 0.0, gpu_elapsed_time_ms = 0.0;
    	const float mytoll = 0.1;

	cl_int error_code = CL_SUCCESS;

	cl_mem bufA = NULL, bufB = NULL, bufC = NULL;
	cl_command_queue queue = NULL;
	cl_context context = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_event event = NULL;
	cl_device_id device = 0;
	cl_platform_id platform = 0;
//------------------------------------- Preparation -----------------------------------------------
    	// Create the matrices and initialize them with random values.
    	float* A = (float*)malloc(M*K*sizeof(float*));
    	float* B = (float*)malloc(K*N*sizeof(float*));
    	float* C = (float*)malloc(M*N*sizeof(float*));
    	float* cpu_C = (float*)malloc(M*N*sizeof(float*));

	if(A == NULL || B == NULL || C == NULL || cpu_C == NULL){
		std::cerr << "Error with malloc()\n";
		goto ErrorAndFree;
	}

    	for (int i = 0; i < M*K; ++i){ 
        	A[i] = static_cast<long long int>(3.6*i + i*i + 3.1) % 256; 
    	}
    	for (int i = 0; i < K*N; ++i){ 
        	B[i] = static_cast<long long int>(1.2*i + 0.01*i*i + 13.9) % 256; 
    	}
    	for (int i = 0; i < M*N; ++i){ 
        	C[i] = cpu_C[i] = 0.0; 
    	}

    	// Configure the OpenCL environment.
   	 std::cout << ">>> Initializing OpenCL...\n";
    	platform = 0;
    	if((error_code = clGetPlatformIDs(1, &platform, NULL)) != CL_SUCCESS){
		std::cerr << "Error with clGetPlatformIDs()\n";
		goto ErrorAndFree;
	}
    	device = 0;
    	if((error_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL)) != CL_SUCCESS){
	    	std::cerr << "Error with clGetDeviceIDs()\n";
	    	goto ErrorAndFree;
	}
    	context = clCreateContext(NULL, 1, &device, NULL, NULL, &error_code);
	if(error_code != CL_SUCCESS){
		std::cerr << "Error with clCreateContext()\n";
		goto ErrorAndFree;
	}
    	//queue = clCreateCommandQueue(context, device, 0, &error_code); // For OpenCL <= 2.x version
    	queue = clCreateCommandQueueWithProperties(context, device, NULL, &error_code); // For OpenCL 3.x version
	if(error_code != CL_SUCCESS){
		std::cerr << "Error with clCreateCommandQueue()\n";
		goto ErrorAndFree;
	}
    	if((error_code = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, NULL)) != CL_SUCCESS){
	    	std::cerr << "Error with clGetDeviceInfo()\n";
	    	goto ErrorAndFree;
	}
    	event = NULL;

    	// Compile the kernel.
   	 program = clCreateProgramWithSource(context, 1, &kernelstring, NULL, &error_code);
	if(error_code != CL_SUCCESS){
		std::cerr << "Error with clCreateProgramWithSource()\n";
		goto ErrorAndFree;
	}
    	if((error_code = clBuildProgram(program, 0, NULL, "", NULL, NULL)) != CL_SUCCESS){
	    	std::cerr << "Error with clBuildProgram()\n";
	    	goto ErrorAndFree;
	}

    	// Check for compilation errors.
    	if((error_code = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size)) != CL_SUCCESS){
	    	std::cerr << "Error with clGetProgramBuildInfo()\n";
	    	goto ErrorAndFree;
	}
    	messages = (char*)malloc((1 + log_size)*sizeof(char));
    	if((error_code = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, messages, NULL)) != CL_SUCCESS){
	    	std::cerr << "Error with clGetProgramBuildInfo()\n";
	    	goto ErrorAndFree;
	}
    	messages[log_size] = '\0';
    	if (log_size > 10) 
        	std::cout << ">>> Compiler message:\n" << messages << "\n";
    
    	free(messages);

    	// Prepare OpenCL memory objects.
    	bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,  M*K*sizeof(float), NULL, &error_code);
	if(error_code != CL_SUCCESS){
        	std::cerr << "Error with clCreateBuffer(1)\n";
       		goto ErrorAndFree;
    	}
    	bufB = clCreateBuffer(context, CL_MEM_READ_ONLY,  K*N*sizeof(float), NULL, &error_code);
	if(error_code != CL_SUCCESS){
        	std::cerr << "Error with clCreateBuffer(2)\n";
       		goto ErrorAndFree;
    	}
    	bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, M*N*sizeof(float), NULL, &error_code);
	if(error_code != CL_SUCCESS){
        	std::cerr << "Error with clCreateBuffer(3)\n";
        	goto ErrorAndFree;
    	}

    	// Copy matrices to the GPU.
    	if((error_code = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, M*K*sizeof(float), A, 0, NULL, NULL)) != CL_SUCCESS){
        	std::cerr << "Error with clEnqueueWriteBuffer(1)\n";
        	goto ErrorAndFree;
    	}
    	if((error_code = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, K*N*sizeof(float), B, 0, NULL, NULL)) != CL_SUCCESS){
        	std::cerr << "Error with clEnqueueWriteBuffer(2)\n";
        	goto ErrorAndFree;
    	}
    	if((error_code = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(float), C, 0, NULL, NULL)) != CL_SUCCESS){
        	std::cerr << "Error with clEnqueueWriteBuffer(3)\n";
        	goto ErrorAndFree;
    	}

    	// Configure the gpu_matrix_mult kernel and set its arguments.
    	kernel = clCreateKernel(program, "gpu_matrix_mult", &error_code);
	if(error_code != CL_SUCCESS){
        	std::cerr << "Error with clCreateKernel()\n";
        	goto ErrorAndFree;
    	}
    	if((error_code = clSetKernelArg(kernel, 0, sizeof(int), (void*)&M)) != CL_SUCCESS){
        	std::cerr << "Error with clSetKernelArg(1)\n";
        	goto ErrorAndFree;
    	}
    	if((error_code = clSetKernelArg(kernel, 1, sizeof(int), (void*)&N)) != CL_SUCCESS){
        	std::cerr << "Error with clSetKernelArg(2)\n";
       		goto ErrorAndFree;
    	}
    	if((error_code = clSetKernelArg(kernel, 2, sizeof(int), (void*)&K)) != CL_SUCCESS){
        	std::cerr << "Error with clSetKernelArg(3)\n";
        	goto ErrorAndFree;
    	}
    	if((error_code = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufA)) != CL_SUCCESS){
        	std::cerr << "Error with clSetKernelArg(4)\n";
        	goto ErrorAndFree;
    	}
    	if((error_code = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufB)) != CL_SUCCESS){
        	std::cerr << "Error with clSetKernelArg(5)\n";
        	goto ErrorAndFree;
    	}
    	if((error_code = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&bufC)) != CL_SUCCESS){
        	std::cerr << "Error with clSetKernelArg(6)\n";
        	goto ErrorAndFree;
    	}
//--------------------------------------- Execution -----------------------------------------------
    	std::cout << ">>> Starting gpu_matrix_mult...\n";
    	// Start the timer for the GPU version.
    	start = std::chrono::high_resolution_clock::now();
    	// Run the gpu_matrix_mult kernel.
    	if((error_code = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event)) != CL_SUCCESS){
        	std::cerr << "Error with clEnqueueNDRangeKernel()\n";
        	goto ErrorAndFree;
    	}
    	// Wait for calculations to be finished.
    	if((error_code = clWaitForEvents(1, &event) != CL_SUCCESS)){
        	std::cerr << "Error with clWaitForEvents()\n";
        	goto ErrorAndFree;
    	}
    	// Stop the timer.
    	end = std::chrono::high_resolution_clock::now();
    	// Copy the output matrix C back to the CPU memory.
    	if((error_code = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(float), C, 0, NULL, NULL)) != CL_SUCCESS){
        	std::cerr << "Error with clEnqueueReadBuffer()\n";
        	goto ErrorAndFree;
    	}
    	// Compute time elapse on GPU computing.
    	gpu_elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    	std::cout << ">>> Time elapsed on matrix multiplication of " << M << "X" << N << " and " << N << "X" << K << " on\n";
	printf("(GPU) : %.3f seconds\n\n", (gpu_elapsed_time_ms/1000.0));

    	std::cout << ">>> Starting cpu_matrix_mult...\n";
    	// Start the timer for the CPU version.
    	start = std::chrono::high_resolution_clock::now();
    	cpu_matrix_mult(A, B, cpu_C, M, N, K);
    	// Stop the timer.
    	end = std::chrono::high_resolution_clock::now();
    	// Compute time elapse on CPU computing.
    	cpu_elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    	std::cout << ">>> Time elapsed on matrix multiplication of " << M << "X" << N << " and " << N << "X" << K << " on\n";
    	printf("(CPU) : %.3f seconds\n\n", (cpu_elapsed_time_ms/1000.0));

    	// Validate the result computed by the GPU.
    	for (int i = 0; i < M; ++i){
        	for (int j = 0; j < K; ++j){
            		if(std::abs(C[i*K + j] - cpu_C[i*K + j]) > mytoll){
                		std::cout << "Error - C[" << i << "*" << K << "+" << j << "] : ";
                		printf("%f", C[i*K + j]); 
                		std::cout << " - cpu_C[" << i << "*" << K << "+" << j << "] : ";
                		printf("%f", cpu_C[i*K + j]);
                		std::cout << " = " << C[i*K + j] - cpu_C[i*K + j] << "\n";
                		/*std::cout << "Matrice A:\n";
                		print_matrix(A, M, K);
                		std::cout << "Matrice B:\n";
                		print_matrix(B, K, N);
                		std::cout << "Matrici C:\n";
                		print_matrix(C, M, N);
                		print_matrix(cpu_C, M, N);*/
                		goto ErrorAndFree;
            		}
        	}
    	}

    	// Roughly compute speedup.
    	std::cout << ">>> GPU performed the multiplication " << static_cast<int>(cpu_elapsed_time_ms / (gpu_elapsed_time_ms + 1.0)) << " times faster than CPU\n";
//---------------------------- Free memory and terminate the program ------------------------------
    	// Free the OpenCL memory objects.
    	clReleaseMemObject(bufA);
    	clReleaseMemObject(bufB);
    	clReleaseMemObject(bufC);
    	// Clean-up OpenCL.
    	clReleaseCommandQueue(queue);
    	clReleaseContext(context);
    	clReleaseProgram(program);
    	clReleaseKernel(kernel);
    	// Free the host memory objects.
    	free(A);
    	free(B);
    	free(C);
    	free(cpu_C);
    	// Exit.
    	return 0;

ErrorAndFree:
	std::cerr << "Something went wrong!\nOpenCL error number: " << error_code << "\n";  
	clReleaseMemObject(bufA);
    	clReleaseMemObject(bufB);
    	clReleaseMemObject(bufC);
    	clReleaseCommandQueue(queue);
    	clReleaseContext(context);
    	clReleaseProgram(program);
    	clReleaseKernel(kernel);
    	free(A);
    	free(B);
    	free(C);
    	free(cpu_C);
	free(messages);
    	return -1;
}
// =================================================================================================
