#ifndef NEURALNET_H
#define NEURALNET_H

#include <curand.h>
#include <curand_kernel.h>

#include <cstdint>
#include <cstdio>

typedef enum {
	NORMAL_NEURON,
	OUTPUT_NEURON,
	INPUT_NEURON,
	NEURON_TYPE_COUNT
} NeuronType;

const char * const NEURON_TYPE_STRINGS[NEURON_TYPE_COUNT] {
	"NORMAL_NEURON",
	"OUTPUT_NEURON",
	"INPUT_NEURON"};

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
	int inputNeuronCount;
	int outputNeuronCount;

	int32_t * h_forwardConnections;
	float * h_connectionWeights;
	float * h_activationThresholds;
	float * h_excitationLevel;
	uint8_t * h_activations;
	uint8_t * h_specialNeurons;
	uint16_t * h_neuronActivationCountRebalance;
	uint16_t * h_neuronActivationCountKilling;

	curandState * d_randState;
	int32_t * d_forwardConnections;
	float * d_connectionWeights;
	float * d_activationThresholds;
	float * d_excitationLevel;
	uint8_t * d_activations;
	uint8_t * d_specialNeurons;
	uint16_t * d_neuronActivationCountRebalance;
	uint16_t * d_neuronActivationCountKilling;

	int32_t * h_inputNeuronIndices;
	uint8_t * h_inputNeuronValues;
	int32_t * h_outputNeuronIndices;
	uint8_t * h_outputNeuronValues;

	int32_t * d_inputNeuronIndices;
	uint8_t * d_inputNeuronValues;
	int32_t * d_outputNeuronIndices;
	uint8_t * d_outputNeuronValues;

	void loadFromFile(FILE * file);
	void allocateAll();
	void copyToCPU();
	void copyToGPU();

public:
	NeuralNet();
	NeuralNet(
		int partitions,
		int neuronsPerPartition,
		int maxConnectionsPerNeuron,
		int feedsBeforeRebalance,
		int rebalancesBeforeKilling,
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
		int outputNeurons,
		int32_t * inputNeuronIndices = 0,
		int32_t * outputNeuronIndices = 0);

	~NeuralNet();

	void randomize();
	void randomizeIONeurons();
	void setInputValues(uint8_t * inputs);
	void feedforward();
	void getOutputValues(uint8_t * outputs);
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
	int getInputNeuronCount();
	int getOutputNeuronCount();

	int32_t * getHostForwardConnections();
	float * getHostConnectionWeights();
	float * getHostActivationThresholds();
	float * getHostExcitationLevel();
	uint8_t * getHostActivations();
	uint8_t * getHostSpecialNeurons();
	uint16_t * getHostNeuronActivationCountRebalance();
	uint16_t * getHostNeuronActivationCountKilling();

	curandState * getDeviceRandState();
	int32_t * getDeviceForwardConnections();
	float * getDeviceConnectionWeights();
	float * getDeviceActivationThresholds();
	float * getDeviceExcitationLevel();
	uint8_t * getDeviceActivations();
	uint8_t * getDeviceSpecialNeurons();
	uint16_t * getDeviceNeuronActivationCountRebalance();
	uint16_t * getDeviceNeuronActivationCountKilling();

	uint8_t * getHostInputNeuronValues();
	int32_t * getHostInputNeuronIndices();
	uint8_t * getHostOutputNeuronValues();
	int32_t * getHostOutputNeuronIndices();

	uint8_t * getDeviceInputNeuronValues();
	int32_t * getDeviceInputNeuronIndices();
	uint8_t * getDeviceOutputNeuronValues();
	int32_t * getDeviceOutputNeuronIndices();
};

#endif