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
	uint16_t minimumKillingActivations;
	uint16_t minimumRebalanceActivations;
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
		uint16_t minimumRebalanceActivations,
		uint16_t minimumKillingActivations,
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

	int getPartitions();
	int getPartitionCount();
	int getNeuronsPerPartition();
	int getMaxConnectionsPerNeuron();
	int getNeuronCount();
	int getConnectionCount();
	int getFeedforwardCount();
	int getFeedsBeforeRebalance();
	int getRebalanceCount();
	int getRebalancesBeforeKilling();
	float getDecayRate();
	float getMinWeightValue();
	float getMaxWeightValue();
	float getMinActivationValue();
	float getMaxActivationValue();
	uint16_t getMinimumKillingActivations();
	uint16_t getMinimumRebalanceActivations();
	float getChangeConstant();
	float getWeightKillValue();
	int getInputNeurons();
	int getOutputNeurons();

	int32_t * getHostForwardConnections();
	float * getHostConnectionWeights();
	float * getHostActivationThresholds();
	float * getHostReceivingSignal();
	float * getHostExcitationLevel();
	uint8_t * getHostActivations();
	uint16_t * getHostNeuronActivationCountRebalance();
	uint16_t * gethHostNeuronActivationCountKilling();

	curandState * getDeviceRandState();
	int32_t * getDeviceForwardConnections();
	float * getDeviceConnectionWeights();
	float * getDeviceActivationThresholds();
	float * getDeviceExcitationLevel();
	uint8_t * getDeviceActivations();
	uint16_t * getDeviceNeuronActivationCountRebalance();
	uint16_t * getDeviceNeuronActivationCountKilling();
};

#endif