#include <stdlib.h>
#include <iostream>
#include "vector_types.h"

#define itemcount 1024*1024

typedef float3 vector;

__global__ void dot(vector *A, vector *B, float *C)
{
	int threadid = threadIdx.x + blockIdx.x*blockDim.x;
	if(threadid < itemcount){
		C[threadid] = 97;//A[threadid].x*B[threadid].x + A[threadid].y*B[threadid].y + A[threadid].z*B[threadid].z;
	}
}

int main()
{
	size_t vecsize = itemcount*sizeof(vector);
	size_t fltsize = itemcount*sizeof(float);
	cudaError_t error;
	vector *A;vector *B;float *C;
	
	cudaHostAlloc(&A, vecsize, cudaHostAllocDefault);
	cudaHostAlloc(&B, vecsize, cudaHostAllocDefault);
	cudaHostAlloc(&C, fltsize, cudaHostAllocDefault);
	
	if(A == NULL || B == NULL || C == NULL)
	{
		std::cout << "Not enough memory on host" << std::endl;
		return -1;
	}
	vector *deviceA;vector *deviceB;float *deviceC;
	cudaMalloc(&deviceA, vecsize);
	cudaMalloc(&deviceB, vecsize);
	cudaMalloc(&deviceC, fltsize);
	
	if(deviceA == NULL || deviceB == NULL || deviceC == NULL)
	{
		std::cout << "Not enough memory on device" << std::endl;
		return -1;
	}
	
	for(int i = 0;i < itemcount;i++)
	{
		C[i] = 42;
		A[i].x = (rand() % 10000 + 1);
		A[i].y = (rand() % 10000 + 1);
		A[i].z = (rand() % 10000 + 1);
		B[i].x = (rand() % 10000 + 1);
		B[i].y = (rand() % 10000 + 1);
		B[i].z = (rand() % 10000 + 1);
	}
	
	error = cudaMemcpy(deviceA, A, vecsize, cudaMemcpyHostToDevice);
	if(cudaSuccess != error) std::cout << "Error: " << cudaGetErrorString(error);
	error = cudaMemcpy(deviceB, B, vecsize, cudaMemcpyHostToDevice);
	if(cudaSuccess != error) std::cout << "Error: " << cudaGetErrorString(error);
	
	
	dim3 threadsPerBlock(1024);
	dim3 blocksPerGrid(1024);
	dot<<<blocksPerGrid, threadsPerBlock>>>(deviceA,deviceB,deviceC);
	
	cudaThreadSynchronize();
	
	error = cudaMemcpy(C, deviceC, fltsize, cudaMemcpyDeviceToHost);
	if(cudaSuccess != error) std::cout << "Error: " << cudaGetErrorString(error);

	std::cout << C[0] << std::endl;
	std::cout << C[56] << std::endl;
	
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
	
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
}