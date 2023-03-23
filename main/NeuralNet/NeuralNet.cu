#include <cstdint>

#include "NeuralNetCUDA.h"
#include "GPUFunctions.h"
#include "NeuralNet.h"

// TODO: Use fp16 to reduce space.
// This should allow roughly double the amount of neurons
// From 3.1MM to 6.5MM

// TODO: Maybe allocating a huge block of memory instead of
// several small blocks is better.

NeuralNet::NeuralNet(
		int partitionsIn,
		int neuronsPerPartitionIn,
		int maxConnectionsPerNeuronIn,
		int feedsBeforeRebalanceIn,
		int rebalancesBeforeKillingIn,
		float decayRateIn,
		float minWeightValueIn,
		float maxWeightValueIn,
		float minActivationValueIn,
		float maxActivationValueIn,
		uint16_t minimumActivationsIn,
		float changeConstantIn,
		float weightKillValueIn,
		int inputNeuronsIn,
		int outputNeuronsIn)
{
	partitions = partitionsIn;
	neuronsPerPartition = neuronsPerPartitionIn;
	maxConnectionsPerNeuron = maxConnectionsPerNeuronIn;
	feedsBeforeRebalance = feedsBeforeRebalanceIn;
	rebalancesBeforeKilling = rebalancesBeforeKillingIn;
	decayRate = decayRateIn;
	minWeightValue = minWeightValueIn;
	maxWeightValue = maxWeightValueIn;
	minActivationValue = minActivationValueIn;
	maxActivationValue = maxActivationValueIn;
	minimumActivations = minimumActivationsIn;
	changeConstant = changeConstantIn;
	weightKillValue = weightKillValueIn;
	inputNeurons = inputNeuronsIn;
	outputNeurons = outputNeuronsIn;

	allocateAll();
}

NeuralNet::~NeuralNet() {
	free(h_forwardConnections);
	free(h_connectionWeights);
	free(h_activationThresholds);
	free(h_receivingSignal);
	free(h_excitationLevel);
	free(h_activations);
	free(h_neuronActivationCountRebalance);
	free(h_neuronActivationCountKilling);

	gpuFree(d_randState);
	gpuFree(d_forwardConnections);
	gpuFree(d_connectionWeights);
	gpuFree(d_activationThresholds);
	gpuFree(d_excitationLevel);
	gpuFree(d_activations);
	gpuFree(d_neuronActivationCountRebalance);
	gpuFree(d_neuronActivationCountKilling);
}


void NeuralNet::allocateAll() {
	partitionCount = partitions * partitions * partitions;
	feedforwardCount = 0;

	// Size: 48 * neuronCount
	// Used for random number generation on the GPU
	d_randState = (curandState *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(curandState));

	// Size: 4 * neuronCount * connectionsPerNeuron
	// List of indices to postsynaptic neurons.
	d_forwardConnections = (int32_t *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(uint32_t));

	// Size: 4 * neuronCount * connectionsPerNeuron
	// Weights to use during feedforward.
	d_connectionWeights = (float *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(float));

	// Size: 4 * neuronCount
	// Activation threshold for each neuron.
	d_activationThresholds = (float *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(float));

	// Size: 4 * neuronCount
	// Current exitation level, when this exceeds the threshold,
	// an activation occurs. This value is set to -1 when the
	// neuron is going to be killed.
	d_excitationLevel = (float *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(float));

	// Size: 1 * neuronCount
	d_activations = (uint8_t *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(uint8_t));


	// Size: 2 * neuronCount
	// Incremented each time a neuron fires. Used to kill unused neurons.
	d_neuronActivationCountRebalance = (uint16_t *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(uint16_t));

	// Size: 2 * neuronCount
	d_neuronActivationCountKilling = (uint16_t *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(uint16_t));

	// TOTAL SIZE: neuronCount * (61 + 8 * connectionsPerNeuron)


	h_forwardConnections = (int32_t *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(int32_t));

	h_connectionWeights = (float *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(float));

	h_activationThresholds = (float *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(float));

	h_receivingSignal = (float *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(float));

	h_excitationLevel = (float *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(float));

	h_neuronActivationCountRebalance = (uint16_t *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(uint16_t));

	h_activations = (uint8_t *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(uint8_t));

	h_neuronActivationCountKilling = (uint16_t *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(uint16_t));
}

void NeuralNet::randomize()
{
	randomizeNeurons(
		d_randState,
		d_activationThresholds,
		minActivationValue,
		maxActivationValue,
		partitionCount,
		neuronsPerPartition);

	createRandomConnections(
		d_randState,
		minWeightValue,
		maxWeightValue,
		d_forwardConnections,
		h_forwardConnections,
		d_connectionWeights,
		partitions,
		partitionCount,
		neuronsPerPartition,
		maxConnectionsPerNeuron,
		inputNeurons);

	normalizeConnections(
		d_forwardConnections,
		d_connectionWeights,
		d_activationThresholds,
		partitionCount * neuronsPerPartition,
		maxConnectionsPerNeuron,
		decayRate);
}

void NeuralNet::feedforward()
{
	mainFeedforward(
		d_excitationLevel,
		d_activations,
		d_forwardConnections,
		d_connectionWeights,
		partitionCount,
		neuronsPerPartition,
		maxConnectionsPerNeuron,
		outputNeurons);

	doExcitationDecay(
		d_excitationLevel,
		decayRate,
		partitionCount,
		neuronsPerPartition,
		maxConnectionsPerNeuron);

	calculateActivations(
		d_excitationLevel,
		d_activationThresholds,
		d_activations,
		d_neuronActivationCountRebalance,
		d_neuronActivationCountKilling,
		partitionCount,
		maxConnectionsPerNeuron);
	
	feedforwardCount++;

	if(feedforwardCount == feedsBeforeRebalance &&
			rebalanceCount == rebalancesBeforeKilling) {
		determineKilledNeurons(
			d_neuronActivationCountKilling,
			d_activations,
			0, // TODO: This needs a real value
			partitionCount,
			neuronsPerPartition);

		randomizeDeadNeurons(
			d_randState,
			minWeightValue,
			maxWeightValue,
			minActivationValue,
			maxActivationValue,
			d_activationThresholds,
			d_forwardConnections,
			h_forwardConnections,
			d_connectionWeights,
			d_activations,
			partitions,
			partitionCount,
			neuronsPerPartition,
			maxConnectionsPerNeuron,
			inputNeurons);

		zeroizeActivationCounts(
			d_neuronActivationCountKilling,
			partitionCount,
			neuronsPerPartition);

		normalizeConnections(
			d_forwardConnections,
			d_connectionWeights,
			d_activationThresholds,
			partitionCount * neuronsPerPartition,
			maxConnectionsPerNeuron,
			decayRate);

		feedforwardCount = 0;
		rebalanceCount = 0;
	} else if(feedforwardCount == feedsBeforeRebalance) {
		rebalanceConnections(
			d_forwardConnections,
			h_forwardConnections,
			d_connectionWeights,
			d_neuronActivationCountRebalance,
			minimumActivations, // TODO: put a real value here
			changeConstant, // TODO: Make these parts of the NN.
			weightKillValue,
			partitions,
			partitionCount,
			neuronsPerPartition,
			maxConnectionsPerNeuron,
			inputNeurons);

		zeroizeActivationCounts(
			d_neuronActivationCountRebalance,
			partitionCount,
			neuronsPerPartition);

		rebalanceCount++;
		feedforwardCount = 0;
	}
}

void NeuralNet::saveToFile(FILE * file) {
	fwrite(&partitionCount, 1, sizeof(int), file);
	fwrite(&neuronsPerPartition, 1, sizeof(int), file);
	fwrite(&maxConnectionsPerNeuron, 1, sizeof(int), file);
	fwrite(&feedforwardCount, 1, sizeof(int), file);
	fwrite(&feedsBeforeRebalance, 1, sizeof(int), file);
	fwrite(&rebalanceCount, 1, sizeof(int), file);
	fwrite(&rebalancesBeforeKilling, 1, sizeof(int), file);
	fwrite(&decayRate, 1, sizeof(int), file);
	fwrite(&minWeightValue, 1, sizeof(float), file);
	fwrite(&maxWeightValue, 1, sizeof(float), file);
	fwrite(&minActivationValue, 1, sizeof(float), file);
	fwrite(&maxActivationValue, 1, sizeof(float), file);
	fwrite(&minimumActivations, 1, sizeof(uint16_t), file);
	fwrite(&changeConstant, 1, sizeof(float), file);
	fwrite(&weightKillValue, 1, sizeof(float), file);
	fwrite(&inputNeurons, 1, sizeof(int), file);
	fwrite(&outputNeurons, 1, sizeof(int), file);

	memcpyGPUtoCPU(h_forwardConnections, 
		d_forwardConnections,
		partitionCount *
			neuronsPerPartition *
			maxConnectionsPerNeuron *
			sizeof(int32_t));

	memcpyGPUtoCPU(h_forwardConnections,
		d_forwardConnections,
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(int32_t));

	memcpyGPUtoCPU(h_connectionWeights,
		d_connectionWeights,
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(float));

	memcpyGPUtoCPU(h_activationThresholds,
		d_activationThresholds,
		partitionCount *
		neuronsPerPartition *
		sizeof(float));

	memcpyGPUtoCPU(h_excitationLevel,
		d_excitationLevel,
		partitionCount *
		neuronsPerPartition *
		sizeof(float));

	memcpyGPUtoCPU(h_neuronActivationCountRebalance,
		d_neuronActivationCountRebalance,
		partitionCount *
		neuronsPerPartition *
		sizeof(uint16_t));

	memcpyGPUtoCPU(h_activations,
		d_activations,
		partitionCount *
		neuronsPerPartition *
		sizeof(uint8_t));

	memcpyGPUtoCPU(h_neuronActivationCountKilling,
		d_neuronActivationCountKilling,
		partitionCount *
		neuronsPerPartition *
		sizeof(uint16_t));


	memcpyGPUtoCPU(h_forwardConnections, 
		d_forwardConnections,
		partitionCount *
			neuronsPerPartition *
			maxConnectionsPerNeuron *
			sizeof(int32_t));

	fwrite(h_forwardConnections,
			partitionCount *
			neuronsPerPartition *
			maxConnectionsPerNeuron,
			sizeof(int32_t), file);

	fwrite(h_connectionWeights,
			partitionCount *
			neuronsPerPartition *
			maxConnectionsPerNeuron,
			sizeof(float), file);

	fwrite(h_activationThresholds,
			partitionCount *
			neuronsPerPartition,
			sizeof(float), file);

	fwrite(h_receivingSignal,
			partitionCount *
			neuronsPerPartition *
			maxConnectionsPerNeuron,
			sizeof(float), file);

	fwrite(h_excitationLevel,
			partitionCount *
			neuronsPerPartition,
			sizeof(float), file);

	fwrite(h_neuronActivationCountRebalance,
			partitionCount *
			neuronsPerPartition,
			sizeof(uint16_t), file);

	fwrite(h_activations,
			partitionCount *
			neuronsPerPartition,
			sizeof(uint8_t), file);

	fwrite(h_neuronActivationCountKilling,
			partitionCount *
			neuronsPerPartition,
			sizeof(uint16_t), file);
}

void NeuralNet::loadFromFile(FILE * file) {
	fread(&partitionCount, 1, sizeof(int), file);
	fread(&neuronsPerPartition, 1, sizeof(int), file);
	fread(&maxConnectionsPerNeuron, 1, sizeof(int), file);
	fread(&feedforwardCount, 1, sizeof(int), file);
	fread(&feedsBeforeRebalance, 1, sizeof(int), file);
	fread(&rebalanceCount, 1, sizeof(int), file);
	fread(&rebalancesBeforeKilling, 1, sizeof(int), file);
	fread(&decayRate, 1, sizeof(int), file);
	fread(&minWeightValue, 1, sizeof(float), file);
	fread(&maxWeightValue, 1, sizeof(float), file);
	fread(&minActivationValue, 1, sizeof(float), file);
	fread(&maxActivationValue, 1, sizeof(float), file);
	fread(&minimumActivations, 1, sizeof(uint16_t), file);
	fread(&changeConstant, 1, sizeof(float), file);
	fread(&weightKillValue, 1, sizeof(float), file);
	fread(&inputNeurons, 1, sizeof(int), file);
	fread(&outputNeurons, 1, sizeof(int), file);

	allocateAll();

	fread(h_forwardConnections,
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron,
		sizeof(int32_t), file);

	fread(h_connectionWeights,
			partitionCount *
			neuronsPerPartition *
			maxConnectionsPerNeuron,
			sizeof(float), file);

	fread(h_activationThresholds,
			partitionCount *
			neuronsPerPartition,
			sizeof(float), file);

	fread(h_receivingSignal,
			partitionCount *
			neuronsPerPartition *
			maxConnectionsPerNeuron,
			sizeof(float), file);

	fread(h_excitationLevel,
			partitionCount *
			neuronsPerPartition,
			sizeof(float), file);

	fread(h_neuronActivationCountRebalance,
			partitionCount *
			neuronsPerPartition,
			sizeof(uint16_t), file);

	fread(h_activations,
			partitionCount *
			neuronsPerPartition,
			sizeof(uint8_t), file);

	fread(h_neuronActivationCountKilling,
			partitionCount *
			neuronsPerPartition,
			sizeof(uint16_t), file);

	memcpyCPUtoGPU(h_forwardConnections,
		d_forwardConnections,
		partitionCount *
			neuronsPerPartition *
			maxConnectionsPerNeuron *
			sizeof(int32_t));

	memcpyCPUtoGPU(h_forwardConnections,
		d_forwardConnections,
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(int32_t));

	memcpyCPUtoGPU(h_connectionWeights,
		d_connectionWeights,
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(float));

	memcpyCPUtoGPU(h_activationThresholds,
		d_activationThresholds,
		partitionCount *
		neuronsPerPartition *
		sizeof(float));

	memcpyGPUtoCPU(h_excitationLevel,
		d_excitationLevel,
		partitionCount *
		neuronsPerPartition *
		sizeof(float));

	memcpyCPUtoGPU(h_neuronActivationCountRebalance,
		d_neuronActivationCountRebalance,
		partitionCount *
		neuronsPerPartition *
		sizeof(uint16_t));

	memcpyCPUtoGPU(h_activations,
		d_activations,
		partitionCount *
		neuronsPerPartition *
		sizeof(uint8_t));

	memcpyCPUtoGPU(h_neuronActivationCountKilling,
		d_neuronActivationCountKilling,
		partitionCount *
		neuronsPerPartition *
		sizeof(uint16_t));

	memcpyCPUtoGPU(h_forwardConnections,
		d_forwardConnections,
		partitionCount *
			neuronsPerPartition *
			maxConnectionsPerNeuron *
			sizeof(int32_t));
}