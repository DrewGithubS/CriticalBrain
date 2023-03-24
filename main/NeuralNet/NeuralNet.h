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
	int neuronCount;
	int connectionCount;
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
	void setInputs(uint8_t * inputs);
	void getoutputs(uint8_t * outputs);

	void getPartitions();
	void getPartitionCount();
	void getNeuronsPerPartition();
	void getMaxConnectionsPerNeuron();
	void getNeuronCount();
	void getConnectionCount();
	void getFeedforwardCount();
	void getFeedsBeforeRebalance();
	void getRebalanceCount();
	void getRebalancesBeforeKilling();
	void getDecayRate();
	void getMinWeightValue();
	void getMaxWeightValue();
	void getMinActivationValue();
	void getMaxActivationValue();
	void getMinimumActivations();
	void getChangeConstant();
	void getWeightKillValue();
	void getInputNeurons();
	void getOutputNeurons();

	void getHostForwardConnections();
	void getHostConnectionWeights();
	void getHostActivationThresholds();
	void getHostReceivingSignal();
	void getHostExcitationLevel();
	void getHostActivations();
	void getHostNeuronActivationCountRebalance();
	void gethHostNeuronActivationCountKilling();

	void getDeviceRandState();
	void getDeviceForwardConnections();
	void getDeviceConnectionWeights();
	void getDeviceActivationThresholds();
	void getDeviceExcitationLevel();
	void getDeviceActivations();
	void getDeviceNeuronActivationCountRebalance();
	void getDeviceNeuronActivationCountKilling();
};

#endif