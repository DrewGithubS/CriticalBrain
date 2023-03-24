#ifndef NEURALNET_H
#define NEURALNET_H

#include <curand.h>
#include <curand_kernel.h>

#include <cstdint>
#include <cstdio>

class NeuralNet {
private:
	int partitions;
	int partitionCount;
	int neuronsPerPartition;
	int maxConnectionsPerNeuron;
	int feedforwardCount;
	int feedsBeforeRebalance;
	int rebalanceCount;
	int rebalancesBeforeKilling;
	float decayRate;
	float minWeightValue;
	float maxWeightValue;
	float minActivationValue;
	float maxActivationValue;
	uint16_t minimumActivations;
	float changeConstant;
	float weightKillValue;
	int inputNeurons;
	int outputNeurons;

	int32_t * h_forwardConnections;
	float * h_connectionWeights;
	float * h_activationThresholds;
	float * h_receivingSignal;
	float * h_excitationLevel;
	uint8_t * h_activations;
	uint16_t * h_neuronActivationCountRebalance;
	uint16_t * h_neuronActivationCountKilling;

	curandState * d_randState;
	int32_t * d_forwardConnections;
	float * d_connectionWeights;
	float * d_activationThresholds;
	float * d_excitationLevel;
	uint8_t * d_activations;
	uint16_t * d_neuronActivationCountRebalance;
	uint16_t * d_neuronActivationCountKilling;

	void loadFromFile(FILE * file);
	void allocateAll();

public:
	NeuralNet();
	NeuralNet(
		int partitions,
		int neuronsPerPartition,
		int maxConnectionsPerNeuron,
		int feedsBeforeRebalanceIn,
		int rebalancesBeforeKillingIn,
		float decayRate,
		float minWeightValue,
		float maxWeightValue,
		float minActivationValue,
		float maxActivationValue,
		uint16_t minimumActivations,
		float changeConstant,
		float weightKillValue,
		int inputNeurons,
		int outputNeurons);

	~NeuralNet();

	void randomize();
	void feedforward();
	void copyToCPU();
	void printNetwork();
	void saveToFile(FILE * file);
};

#endif