#include <iostream>

void * gpuMemAlloc(uint32_t bytes) {
	void * output;
	cudaError_t err = cudaMalloc(&output, bytes);
	if ( err != cudaSuccess ) {
		std::cout << cudaGetErrorString(err) << std::endl;
		return NULL;
	}

	return output;
};