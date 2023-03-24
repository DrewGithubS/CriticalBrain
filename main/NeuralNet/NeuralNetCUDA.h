#ifndef NEURALNETCUDA_H
#define NEURALNETCUDA_H

#include <curand.h>
#include <curand_kernel.h>

void setupRand(
	curandState * curandStates,
	int seed,
	int16_t partitions,
	int neuronsPerPartition);

void randomizeNeurons(
	curandState * curandStates,
	float * activationThresholds,
	float minValue,
	float maxValue,
	int16_t partitions,
	int neuronsPerPartition);

void createRandomConnections(
	curandState * curandStates,
	float minWeight,
	float maxWeight,
	int32_t * d_forwardConnections,
	int32_t * h_forwardConnections,
	float * connectionWeights,
	int partitions,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron,
	int inputNeurons);

void normalizeConnections(
	int32_t * forwardConnections,
	float * connectionWeights,
	float * activationThresholds,
	int neurons,
	int connectionsPerNeuron,
	float decayRate);

void mainFeedforward(
	float * excitationLevel,
	uint8_t * activations,
	int32_t * forwardConnections,
	float * connectionWeights,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron,
	int outputNeurons);

void doExcitationDecay(
	float * excitationLevel,
	float decayRate,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron);

void calculateActivations(
	float * excitationLevel,
	float * activationThresholds,
	uint8_t * activations,
	uint16_t * activationCount1,
	uint16_t * activationCount2,
	int partitionCount,
	int neuronsPerPartition);

void determineKilledNeurons(
	uint16_t * activationCount,
	uint8_t * activations,
	uint16_t minimumActivations,
	int partitionCount,
	int neuronsPerPartition);

void randomizeDeadNeurons(
	curandState * curandStates,
	float minWeight,
	float maxWeight,
	float minActivation,
	float maxActivation,
	float * activationThresholds,
	int32_t * d_forwardConnections,
	int32_t * h_forwardConnections,
	float * connectionWeights,
	uint8_t * activations,
	int partitions,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron,
	int inputNeurons);

void zeroizeActivationCounts(
	uint16_t * activationCount,
	int partitionCount,
	int neuronsPerPartition);

void rebalanceConnections(
	int32_t * d_forwardConnections,
	int32_t * h_forwardConnections,
	float * connectionWeights,
	uint16_t * activationCount,
	uint16_t minimumActivations,
	float changeConstant,
	float minimumWeightValue,
	int partitions,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron,
	int inputNeurons);

#endif