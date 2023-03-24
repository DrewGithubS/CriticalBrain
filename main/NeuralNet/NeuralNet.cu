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
		uint16_t minimumKillingActivationsIn,
		uint16_t minimumRebalanceActivationsIn,
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
	minimumKillingActivations = minimumKillingActivationsIn;
	minimumRebalanceActivations = minimumRebalanceActivationsIn;
	changeConstant = changeConstantIn;
	weightKillValue = weightKillValueIn;
	inputNeurons = inputNeuronsIn;
	outputNeurons = outputNeuronsIn;

	allocateAll();

	setupRand(
		this,
		rand());
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
	rebalanceCount = 0;
	neuronCount = partitionCount * neuronsPerPartition;
	connectionCount = neuronCount * maxConnectionsPerNeuron;

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


	h_forwardConnections = (int32_t *) malloc(
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(int32_t));

	h_connectionWeights = (float *) malloc(
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(float));

	h_activationThresholds = (float *) malloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(float));

	h_receivingSignal = (float *) malloc(
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(float));

	h_excitationLevel = (float *) malloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(float));

	h_neuronActivationCountRebalance = (uint16_t *) malloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(uint16_t));

	h_activations = (uint8_t *) malloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(uint8_t));

	h_neuronActivationCountKilling = (uint16_t *) malloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(uint16_t));
}

void NeuralNet::randomize()
{
	printf("Randomizing neurons...\n"); fflush(stdout);
	randomizeNeurons(
		this);

	printf("Randomizing conenctions...\n"); fflush(stdout);
	createRandomConnections(
		this);

	printf("Normalizing conenctions...\n"); fflush(stdout);
	normalizeConnections(
		this);

	zeroizeActivationCounts(
		d_neuronActivationCountKilling,
		neuronCount);

	zeroizeActivationCounts(
		d_neuronActivationCountRebalance,
		neuronCount);
}

void NeuralNet::feedforward()
{
	mainFeedforward(
		this);

	doExcitationDecay(
		this);

	calculateActivations(
		this);
	
	feedforwardCount++;

	if(feedforwardCount == feedsBeforeRebalance &&
			rebalanceCount == rebalancesBeforeKilling) {
		determineKilledNeurons(
			this);

		randomizeDeadNeurons(
			this);

		zeroizeActivationCounts(
			d_neuronActivationCountKilling,
			neuronCount);

		normalizeConnections(
			this);

		feedforwardCount = 0;
		rebalanceCount = 0;
	} else if(feedforwardCount == feedsBeforeRebalance) {
		rebalanceConnections(
			this);

		zeroizeActivationCounts(
			d_neuronActivationCountRebalance,
			neuronCount);

		rebalanceCount++;
		feedforwardCount = 0;
	}
}

void NeuralNet::copyToCPU() {
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
}

void NeuralNet::printNetwork() {
	copyToCPU();

	for(int i = 0; i < neuronsPerPartition * partitionCount; i++) {
		printf("Neuron %d:\n", i); fflush(stdout);
		printf("\ti: %d\n", i); fflush(stdout);
		printf("\tExcitation Level: %f\n", h_excitationLevel[i]); fflush(stdout);
		printf("\tActivation Threshold: %f\n", h_activationThresholds[i]); fflush(stdout);
		printf("\tActivated: %d\n", h_activations[i]); fflush(stdout);
		printf("\tConnections:\n"); fflush(stdout);
		for(int j = 0; j < maxConnectionsPerNeuron; j++) {
			printf("\t\tConnection : %d\n", h_forwardConnections[
				i * maxConnectionsPerNeuron + j]); fflush(stdout);
			printf("\t\tWeight : %f\n", h_connectionWeights[
				i * maxConnectionsPerNeuron + j]); fflush(stdout);
		}
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
	fwrite(&minimumKillingActivations, 1, sizeof(uint16_t), file);
	fwrite(&minimumRebalanceActivations, 1, sizeof(uint16_t), file);
	fwrite(&changeConstant, 1, sizeof(float), file);
	fwrite(&weightKillValue, 1, sizeof(float), file);
	fwrite(&inputNeurons, 1, sizeof(int), file);
	fwrite(&outputNeurons, 1, sizeof(int), file);

	copyToCPU();

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
	fread(&minimumKillingActivations, 1, sizeof(uint16_t), file);
	fread(&minimumRebalanceActivations, 1, sizeof(uint16_t), file);
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





/********************************** GETTERS *********************************/

int NeuralNet::getPartitions() {
	return partitions;
}

int NeuralNet::getPartitionCount() {
	return partitionCount;
}

int NeuralNet::getNeuronsPerPartition() {
	return neuronsPerPartition;
}

int NeuralNet::getMaxConnectionsPerNeuron() {
	return maxConnectionsPerNeuron;
}

int NeuralNet::getNeuronCount() {
	return neuronCount;
}

int NeuralNet::getConnectionCount() {
	return connectionCount;
}

int NeuralNet::getFeedforwardCount() {
	return feedforwardCount;
}

int NeuralNet::getFeedsBeforeRebalance() {
	return feedsBeforeRebalance;
}

int NeuralNet::getRebalanceCount() {
	return rebalanceCount;
}

int NeuralNet::getRebalancesBeforeKilling() {
	return rebalancesBeforeKilling;
}

float NeuralNet::getDecayRate() {
	return decayRate;
}

float NeuralNet::getMinWeightValue() {
	return minWeightValue;
}

float NeuralNet::getMaxWeightValue() {
	return maxWeightValue;
}

float NeuralNet::getMinActivationValue() {
	return minActivationValue;
}

float NeuralNet::getMaxActivationValue() {
	return maxActivationValue;
}

uint16_t NeuralNet::getMinimumKillingActivations() {
	return minimumKillingActivations;
}

uint16_t NeuralNet::getMinimumRebalanceActivations() {
	return minimumRebalanceActivations;
}

float NeuralNet::getChangeConstant() {
	return changeConstant;
}

float NeuralNet::getWeightKillValue() {
	return weightKillValue;
}

int NeuralNet::getInputNeurons() {
	return inputNeurons;
}

int NeuralNet::getOutputNeurons() {
	return outputNeurons;
}


int32_t * NeuralNet::getHostForwardConnections() {
	return h_forwardConnections;
}

float * NeuralNet::getHostConnectionWeights() {
	return h_connectionWeights;
}

float * NeuralNet::getHostActivationThresholds() {
	return h_activationThresholds;
}

float * NeuralNet::getHostReceivingSignal() {
	return h_receivingSignal;
}

float * NeuralNet::getHostExcitationLevel() {
	return h_excitationLevel;
}

uint8_t * NeuralNet::getHostActivations() {
	return h_activations;
}

uint16_t * NeuralNet::getHostNeuronActivationCountRebalance() {
	return h_neuronActivationCountRebalance;
}

uint16_t * NeuralNet::gethHostNeuronActivationCountKilling() {
	return h_neuronActivationCountKilling;
}


curandState * NeuralNet::getDeviceRandState() {
	return d_randState;
}

int32_t * NeuralNet::getDeviceForwardConnections() {
	return d_forwardConnections;
}

float * NeuralNet::getDeviceConnectionWeights() {
	return d_connectionWeights;
}

float * NeuralNet::getDeviceActivationThresholds() {
	return d_activationThresholds;
}

float * NeuralNet::getDeviceExcitationLevel() {
	return d_excitationLevel;
}

uint8_t * NeuralNet::getDeviceActivations() {
	return d_activations;
}

uint16_t * NeuralNet::getDeviceNeuronActivationCountRebalance() {
	return d_neuronActivationCountRebalance;
}

uint16_t * NeuralNet::getDeviceNeuronActivationCountKilling() {
	return d_neuronActivationCountKilling;
}
