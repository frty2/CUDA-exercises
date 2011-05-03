#include <iostream>
#include <glog/logging.h>

#include "copyprimitives.h"


void copyPrimitives(const primitives &objects)
{
    std::cout << "copying primitives" << std::endl;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaError_t error;
	triangle *deviceobjects;
	size_t sizeinbytes = objects.count*sizeof(triangle);
	
	error = cudaMalloc(&deviceobjects, sizeinbytes);
	CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
	
	cudaEventRecord(start, 0);
	error = cudaMemcpy(deviceobjects, objects.triangles, sizeinbytes, cudaMemcpyHostToDevice);
	CHECK_EQ(cudaSuccess, error) << "Error: " << cudaGetErrorString(error);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	std::cout << elapsedTime << " ms elapsed for copy operation" << std::endl;
}