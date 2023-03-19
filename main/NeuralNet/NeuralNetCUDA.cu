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

/*
          zyx
0  1  2   000 001 002
3  4  5   010 011 012
6  7  8   020 021 022

9  10 11  100 101 102
12 13 14  110 111 112
15 16 17  120 121 122

18 19 20  200 201 202
21 22 23  210 211 212
24 25 26  220 221 222

x = (n % 3)
y = ((n / 3) % 3)
z = (n / (3 * 3))
*/

__global__ void d_createRandomConnections(curandState * curandStates,
										  float minWeight,
										  float maxWeight,
										  int32_t * forwardConnections,
										  int32_t * connectionsWeights,
								   		  int partitions,
								   		  int neuronsPerPartition,
								   		  int neuronCount,
								   		  int connectionsPerNeuron,
								   		  int connectionCount) {

	int index = threadIdx.x + blockDim.x*blockIdx.x;

	if(index < connectionCount) {
		int neuronIndex = index / connectionsPerNeuron;
		int partitionIndex = neuronIndex / neuronsPerPartition;
		int partitionX = (partitionIndex % partitions); 
		int partitionY = (partitionIndex / partitions) % partitions;
		int partitionZ = (partitionIndex / (partitions * partitions));
		
		float minValueX = partitionX == 0 ? 0 : -1;
		float minValueY = partitionY == 0 ? 0 : -1;
		float minValueZ = partitionZ == 0 ? 0 : -1;

		float maxValueX = partitionX == (partitions - 1) ? 0 : 1;
		float maxValueY = partitionY == (partitions - 1) ? 0 : 1;
		float maxValueZ = partitionZ == (partitions - 1) ? 0 : 1;

		

		float x_f = curand_uniform(curandStates + index);
	    x_f *= (maxValueX - minValueX + 0.999999);
	    x_f += minValueX;
	    int dx = (int) truncf(x_f);

	    float y_f = curand_uniform(curandStates + index);
	    y_f *= (maxValueY - minValueY + 0.999999);
	    y_f += maxValueY;
	    int dy = (int) truncf(y_f);

	    float z_f = curand_uniform(curandStates + index);
	    z_f *= (maxValueZ - minValueZ + 0.999999);
	    z_f += maxValueZ;
	    int dz = (int) truncf(z_f);

	    int neuronPartitionIndex = 
	    		(partitionZ + dz) * partitions * partitions +
				(partitionY + dy) * partitions + 
				(partitionX + dx);

        neuronPartitionIndex *= neuronsPerPartition;

		int newNeuronIndex;
		do {
			float z_f = curand_uniform(curandStates + index);
		    z_f *= (neuronsPerPartition - 0 + 0.999999);
		    z_f += 0;
		    newNeuronIndex = neuronPartitionIndex + (int) truncf(z_f);
		} while(newNeuronIndex == neuronIndex);

		forwardConnections[index] = neuronIndex;

		float weight = curand_uniform(curandStates + index);
	    weight *= (maxWeight - minWeight + 0.999999);
	    weight += minWeight;
		
	    connectionsWeights[index] = weight;
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
							 float minWeight,
							 float maxWeight,
							 int32_t * forwardConnections,
							 int32_t * connectionsWeights,
							 int partitions,
							 int neuronsPerPartition,
							 int neuronCount,
							 int connectionsPerNeuron,
							 int connectionCount) {

	int neurons = partitionCount * neuronsPerPartition;
	int connections = neurons * maxConnectionsPerNeuron;

	d_createRandomConnections <<< 
								  BlockCount(connections),
								  THREADSPERBLOCK >>> (
		curandStates,
		minWeight,
		maxWeight,
		forwardConnections,
		connectionsWeights,
		partitions,
		neuronsPerPartition,
		neuronCount,
		connectionsPerNeuron,
		connectionCount);
}