#include <cstdint>

#include "NeuralNetCUDA.h"

const uint32_t THREADSPERBLOCK = 1024;	
#define BlockCount(x) ((x + THREADSPERBLOCK - 1)/THREADSPERBLOCK)


__global__ void d_setupRand(curandState *state,
						    int neurons) {

	int index = threadIdx.x + blockDim.x * blockIdx.x;
	
	if(index < neurons) {
		curand_init(1234, index, 0, &state[index]);
	}
}

__global__ void d_randomizeNeurons(curandState * curandStates,
								   float * activationThresholds,
								   int16_t * partitionLocs,
								   float minValue,
								   float maxValue,
								   int partitions,
								   int neurons){

	int index = threadIdx.x + blockDim.x*blockIdx.x;

	if(index < neurons) {
		activationThresholds[index] = curand_uniform(curandStates + index);
		activationThresholds[index] *= (maxValue - minValue + 0.999999);
		activationThresholds[index] += minValue;
		partitionLocs[index] = index/partitions;
	}
}

__global__ void d_createRandomConnections(curandState * curandStates,
								   		  float minValue,
								   		  float maxValue,
								   		  int partitions,
								   		  int connections){

	int index = threadIdx.x + blockDim.x*blockIdx.x;

	if(index < connections) {
		
	}
}

void randomizeNeurons(curandState * curandStates,
					  float * activationThresholds,
					  int16_t * partitionLocs,
					  float minValue,
					  float maxValue,
					  int16_t partitions,
					  int neuronsPerPartition) {

	int neurons = partitions * neuronsPerPartition;

	d_setupRand <<< BlockCount(neurons), THREADSPERBLOCK >>> (
		curandStates,
		neurons);

	d_randomizeNeurons <<< BlockCount(neurons), THREADSPERBLOCK >>> (
		curandStates,
		activationThresholds,
		partitionLocs,
		minValue,
		maxValue,
		partitions,
		neurons);
}

void createRandomConnections(curandState * curandStates,
							 int16_t * partitionLocs,
							 int32_t * forwardConnections,
							 float * forwardConnectionWeights,
							 int partitionCount,
							 int neuronsPerPartition,
							 int maxConnectionsPerNeuron) {

	int connections = partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron;

	// d_createRandomConnections <<< 
	// 							  BlockCount(connections),
	// 							  THREADSPERBLOCK >>> (
	// 	curandStates,
	// 	activationThresholds,
	// 	minValue,
	// 	maxValue,
	// 	partitions,
	// 	partitions * neuronsPerPartition);
}