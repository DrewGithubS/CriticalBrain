#ifndef NEURALNETCUDA_H
#define NEURALNETCUDA_H

#include <curand.h>
#include <curand_kernel.h>

void randomizeNeurons(curandState * curandStates,
					  float * activationThresholds,
					  float minValue,
					  float maxValue,
					  int16_t partitions,
					  int neuronsPerPartition);

void createRandomConnections(curandState * curandStates,
							 float minWeight,
							 float maxWeight,
							 int32_t * forwardConnections,
							 float * connectionWeights,
							 int partitions,
							 int partitionCount,
							 int neuronsPerPartition,
							 int connectionsPerNeuron);

void normalizeConnections();

#endif