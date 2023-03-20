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

	int32_t * h_forwardConnections;
	float * h_forwardConnectionWeights;
	float * h_activationThresholds;
	float * h_receivingSignal;
	float * h_excitationLevel;
	uint8_t * h_activations;
	uint16_t * h_neuronActivationCountRebalance;
	uint16_t * h_neuronActivationCountKilling;

	curandState * d_randState;
	int32_t * d_forwardConnections;
	float * d_forwardConnectionWeights;
	float * d_activationThresholds;
	float * d_receivingSignal;
	float * d_excitationLevel;
	uint8_t * d_activations;
	uint16_t * d_neuronActivationCountRebalance;
	uint16_t * d_neuronActivationCountKilling;
public:
	NeuralNet(int partitions,
			  int neuronsPerPartition,
			  int maxConnectionsPerNeuron,
			  int feedsBeforeRebalanceIn,
			  int rebalancesBeforeKillingIn);

	void randomize();
	void feedforward();
	void saveToFile(FILE * file);
	void loadFromFile(FILE * file);
};

#endif