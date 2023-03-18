#ifndef NEURALNETCUDA_H
#define NEURALNETCUDA_H

#include <curand.h>
#include <curand_kernel.h>

void randomizeNeurons(curandState * curandStates,
					  float * activationThresholds,
					  int16_t * partitionLocs,
					  float minValue,
					  float maxValue,
					  int16_t partitions,
					  int neuronsPerPartition);

void createRandomConnections(curandState * curandStates,
							 int16_t * partitionLocs,
							 int32_t * forwardConnections,
							 float * forwardConnectionWeights,
							 int partitionCount,
							 int neuronsPerPartition,
							 int maxConnectionsPerNeuron);

#endif