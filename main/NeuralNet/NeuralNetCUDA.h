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
							 int32_t * d_forwardConnections,
							 int16_t * d_forwardConnectionsSub,
							 int32_t * h_forwardConnections,
							 int16_t * h_forwardConnectionsSub,
							 int16_t * h_tempNeuronConnectionSub,
							 float * connectionWeights,
							 int partitions,
							 int partitionCount,
							 int neuronsPerPartition,
							 int connectionsPerNeuron);

void normalizeConnections(int32_t * forwardConnections,
						  float * connectionWeights,
						  float * activationThresholds,
						  int neurons,
						  int connectionsPerNeuron,
						  float decayRate);

void zeroizeReceivers(float * receivingSignal,
					  int partitionCount,
					  int neuronsPerPartition,
					  int connectionsPerNeuron);

void mainFeedforward(float * receivingSignal,
					 uint8_t * activations,
					 int32_t * forwardConnections,
					 int16_t * forwardConnectionsSub,
					 float * connectionWeights,
					 int partitionCount,
					 int neuronsPerPartition,
					 int connectionsPerNeuron);

void doNeuronReduction(float * receivingSignal,
					   int partitionCount,
					   int neuronsPerPartition,
					   int connectionsPerNeuron);

void doExcitationDecay(float * receivingSignal,
					   float * excitationLevel,
					   float decayRate,
					   int partitionCount,
					   int neuronsPerPartition,
					   int connectionsPerNeuron);

void calculateActivations(float * excitationLevel,
						  float * activationThresholds,
						  uint8_t * activations,
						  uint16_t * activationCount1,
						  uint16_t * activationCount2,
						  int partitionCount,
						  int neuronsPerPartition);

void determineKilledNeurons(uint16_t * activationCount,
							uint8_t * activations,
							uint16_t minimumActivations,
							int partitionCount,
							int neuronsPerPartition);

#endif