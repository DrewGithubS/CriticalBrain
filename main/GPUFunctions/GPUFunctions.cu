#include <iostream>

#include "GPUFunctions.h"

void * gpuMemAlloc(uint32_t bytes) {
	void * output;
	cudaError_t err = cudaMalloc(&output, bytes);
	if ( err != cudaSuccess ) {
		std::cout << cudaGetErrorString(err) << std::endl;
		return NULL;
	}

	return output;
};

void memcpyCPUtoGPU(void * to, void * from, size_t size) {
	cudaMemcpy(to, from, size, cudaMemcpyHostToDevice);
}

void memcpyGPUtoCPU(void * to, void * from, size_t size) {
	cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost);
}