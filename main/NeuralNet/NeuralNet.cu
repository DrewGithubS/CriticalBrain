#include <cstdint>

#include "NeuralNetCUDA.h"
#include "GPUFunctions.h"
#include "NeuralNet.h"

// TODO: Use fp16 to reduce space.
// This should allow roughly double the amount of neurons
// From 3.1MM to 6.5MM

// TODO: Maybe allocating a huge block of memory instead of
// several small blocks is better.

bool arrayContains(
	int32_t * array,
	int32_t value,
	int index) {

	for(int i = 0; i < index; i++) {
		if(array[i] == value) {
			return 1;
		}
	}

	return 0;
}

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
		int inputNeuronCountIn,
		int outputNeuronCountIn,
		int32_t * inputNeuronIndicesIn,
		int32_t * outputNeuronIndicesIn)
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
	inputNeuronCount = inputNeuronCountIn;
	outputNeuronCount = outputNeuronCountIn;

	allocateAll();

	if(inputNeuronIndicesIn != 0 && outputNeuronIndicesIn != 0) {
		memcpy(h_inputNeuronIndices,
			inputNeuronIndicesIn,
			inputNeuronCount * sizeof(int32_t));

		memcpyCPUtoGPU(d_outputNeuronIndices,
			h_outputNeuronIndices,
			outputNeuronCount *
			sizeof(int32_t *));

		memcpy(h_outputNeuronIndices,
			outputNeuronIndicesIn,
			outputNeuronCount * sizeof(int32_t));

		memcpyGPUtoCPU(d_outputNeuronIndices,
			h_outputNeuronIndices,
			outputNeuronCount *
			sizeof(int32_t *));

		setSpecialNeurons(this);
	}

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
		neuronCount *
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
		connectionCount *
		sizeof(float));

	// Size: 4 * neuronCount
	// Activation threshold for each neuron.
	d_activationThresholds = (float *) gpuMemAlloc(
		neuronCount *
		sizeof(float));

	// Size: 4 * neuronCount
	// Current exitation level, when this exceeds the threshold,
	// an activation occurs. This value is set to -1 when the
	// neuron is going to be killed.
	d_excitationLevel = (float *) gpuMemAlloc(
		neuronCount *
		sizeof(float));

	// Size: 1 * neuronCount
	d_activations = (uint8_t *) gpuMemAlloc(
		neuronCount *
		sizeof(uint8_t));

	// Size: 2 * neuronCount
	// Incremented each time a neuron fires. Used to kill unused neurons.
	d_specialNeurons = (uint8_t *) gpuMemAlloc(
		neuronCount *
		sizeof(uint16_t));

	// Size: 2 * neuronCount
	// Incremented each time a neuron fires. Used to kill unused neurons.
	d_neuronActivationCountRebalance = (uint16_t *) gpuMemAlloc(
		neuronCount *
		sizeof(uint16_t));

	// Size: 2 * neuronCount
	d_neuronActivationCountKilling = (uint16_t *) gpuMemAlloc(
		neuronCount *
		sizeof(uint16_t));

	// TOTAL SIZE: neuronCount * (61 + 8 * connectionsPerNeuron)


	h_forwardConnections = (int32_t *) malloc(
		connectionCount *
		sizeof(int32_t));

	h_connectionWeights = (float *) malloc(
		connectionCount *
		sizeof(float));

	h_activationThresholds = (float *) malloc(
		neuronCount *
		sizeof(float));

	h_receivingSignal = (float *) malloc(
		connectionCount *
		sizeof(float));

	h_excitationLevel = (float *) malloc(
		neuronCount *
		sizeof(float));

	h_activations = (uint8_t *) malloc(
		neuronCount *
		sizeof(uint8_t));

	h_specialNeurons = (uint8_t *) malloc(
		neuronCount *
		sizeof(uint16_t));

	h_neuronActivationCountRebalance = (uint16_t *) malloc(
		neuronCount *
		sizeof(uint16_t));

	h_neuronActivationCountKilling = (uint16_t *) malloc(
		neuronCount *
		sizeof(uint16_t));



	h_inputNeuronIndices = (int32_t *) malloc(
		inputNeuronCount *
		sizeof(int32_t));

	h_inputNeuronValues = (uint8_t *) malloc(
		inputNeuronCount *
		sizeof(uint8_t));

	h_outputNeuronIndices = (int32_t *) malloc(
		outputNeuronCount *
		sizeof(int32_t));

	h_outputNeuronValues = (uint8_t *) malloc(
		outputNeuronCount *
		sizeof(uint8_t));


	d_inputNeuronIndices = (int32_t *) gpuMemAlloc(
		inputNeuronCount *
		sizeof(int32_t));

	d_inputNeuronValues = (uint8_t *) gpuMemAlloc(
		inputNeuronCount *
		sizeof(uint8_t));

	d_outputNeuronIndices = (int32_t *) gpuMemAlloc(
		outputNeuronCount *
		sizeof(int32_t));

	d_outputNeuronValues = (uint8_t *) gpuMemAlloc(
		outputNeuronCount *
		sizeof(uint8_t));
}

void NeuralNet::randomize()
{
	struct timespec start;
	struct timespec stop;

	clock_gettime(CLOCK_REALTIME, &start);
	randomizeNeurons(
		this);
	clock_gettime(CLOCK_REALTIME, &stop);
	uint64_t accum = ( stop.tv_sec - start.tv_sec ) * 1000000000 +
		( stop.tv_nsec - start.tv_nsec );

    printf("\tTime taken to randomize neurons: %llu.%llu seconds...\n",
    	accum / 1000000000,
    	accum % 1000000000);

	clock_gettime(CLOCK_REALTIME, &start);
	createRandomConnections(
		this);
	clock_gettime(CLOCK_REALTIME, &stop);
	accum = ( stop.tv_sec - start.tv_sec ) * 1000000000 +
		( stop.tv_nsec - start.tv_nsec );

    printf("\tTime taken to randomize connections: %llu.%llu seconds...\n",
    	accum / 1000000000,
    	accum % 1000000000);

	clock_gettime(CLOCK_REALTIME, &start);
	normalizeConnections(
		this);
	clock_gettime(CLOCK_REALTIME, &stop);
	accum = ( stop.tv_sec - start.tv_sec ) * 1000000000 +
		( stop.tv_nsec - start.tv_nsec );
	printf("\tTime taken to normalize connections: %llu.%llu seconds...\n",
    	accum / 1000000000,
    	accum % 1000000000);

	zeroizeActivationCounts(
		d_neuronActivationCountKilling,
		neuronCount);

	zeroizeActivationCounts(
		d_neuronActivationCountRebalance,
		neuronCount);
}

void NeuralNet::randomizeIONeurons() {
	for(int i = 0; i < inputNeuronCount; i++) {
		h_inputNeuronIndices[i] = rand() % neuronCount;
		while(arrayContains(
				h_inputNeuronIndices,
				h_inputNeuronIndices[i],
				i)) {

			h_inputNeuronIndices[i] = rand() % neuronCount;
		}
	}

	for(int i = 0; i < inputNeuronCount; i++) {
		h_outputNeuronIndices[i] = rand() % neuronCount;
		while(arrayContains(
				h_inputNeuronIndices,
				h_inputNeuronIndices[i],
				i) ||
			  arrayContains(
				h_inputNeuronIndices,
				h_outputNeuronIndices[i],
				inputNeuronCount)) {

			h_outputNeuronIndices[i] = rand() % neuronCount;
		}
	}

	memcpyCPUtoGPU(d_outputNeuronIndices,
			h_outputNeuronIndices,
			outputNeuronCount *
			sizeof(int32_t *));

	memcpyGPUtoCPU(d_outputNeuronIndices,
		h_outputNeuronIndices,
		outputNeuronCount *
		sizeof(int32_t *));

	setSpecialNeurons(this);
}


void NeuralNet::setInputValues(uint8_t * inputs) {
	memcpy(h_inputNeuronValues, inputs, inputNeuronCount * sizeof(uint8_t *));

	setNetworkInputs(this);
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

void NeuralNet::getOutputValues(uint8_t * outputs) {
	getNetworkOutputs(this);

	memcpy(outputs, h_outputNeuronValues, outputNeuronCount * sizeof(uint8_t *));
}

void NeuralNet::copyToCPU() {
	memcpyGPUtoCPU(h_forwardConnections, 
		d_forwardConnections,
		connectionCount *
		sizeof(int32_t));

	memcpyGPUtoCPU(h_connectionWeights,
		d_connectionWeights,
		connectionCount *
		sizeof(float));

	memcpyGPUtoCPU(h_activationThresholds,
		d_activationThresholds,
		neuronCount *
		sizeof(float));

	memcpyGPUtoCPU(h_excitationLevel,
		d_excitationLevel,
		neuronCount *
		sizeof(float));

	memcpyGPUtoCPU(h_neuronActivationCountRebalance,
		d_neuronActivationCountRebalance,
		neuronCount *
		sizeof(uint16_t));

	memcpyGPUtoCPU(h_activations,
		d_activations,
		neuronCount *
		sizeof(uint8_t));

	memcpyGPUtoCPU(h_neuronActivationCountKilling,
		d_neuronActivationCountKilling,
		neuronCount *
		sizeof(uint16_t));

	memcpyGPUtoCPU(h_forwardConnections, 
		d_forwardConnections,
		connectionCount *
		sizeof(int32_t));


	memcpyGPUtoCPU(h_inputNeuronIndices,
		d_inputNeuronIndices,
		inputNeuronCount *
		sizeof(int32_t *));

	memcpyGPUtoCPU(h_inputNeuronValues,
		d_inputNeuronValues,
		inputNeuronCount *
		sizeof(uint8_t *));

	memcpyGPUtoCPU(h_outputNeuronIndices,
		d_outputNeuronIndices,
		outputNeuronCount *
		sizeof(int32_t *));

	memcpyGPUtoCPU(h_outputNeuronValues,
		d_outputNeuronValues,
		outputNeuronCount *
		sizeof(uint8_t *));
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
	fwrite(&inputNeuronCount, 1, sizeof(int), file);
	fwrite(&outputNeuronCount, 1, sizeof(int), file);

	copyToCPU();

	fwrite(h_forwardConnections,
			connectionCount,
			sizeof(int32_t), file);

	fwrite(h_connectionWeights,
			connectionCount,
			sizeof(float), file);

	fwrite(h_activationThresholds,
			neuronCount,
			sizeof(float), file);

	fwrite(h_receivingSignal,
			connectionCount,
			sizeof(float), file);

	fwrite(h_excitationLevel,
			neuronCount,
			sizeof(float), file);

	fwrite(h_neuronActivationCountRebalance,
			neuronCount,
			sizeof(uint16_t), file);

	fwrite(h_activations,
			neuronCount,
			sizeof(uint8_t), file);

	fwrite(h_neuronActivationCountKilling,
			neuronCount,
			sizeof(uint16_t), file);
}

void NeuralNet::copyToGPU() {
		memcpyCPUtoGPU(h_forwardConnections,
		d_forwardConnections,
		connectionCount *
		sizeof(int32_t));

	memcpyCPUtoGPU(h_forwardConnections,
		d_forwardConnections,
		connectionCount *
		sizeof(int32_t));

	memcpyCPUtoGPU(h_connectionWeights,
		d_connectionWeights,
		connectionCount *
		sizeof(float));

	memcpyCPUtoGPU(h_activationThresholds,
		d_activationThresholds,
		neuronCount *
		sizeof(float));

	memcpyGPUtoCPU(h_excitationLevel,
		d_excitationLevel,
		neuronCount *
		sizeof(float));

	memcpyCPUtoGPU(h_neuronActivationCountRebalance,
		d_neuronActivationCountRebalance,
		neuronCount *
		sizeof(uint16_t));

	memcpyCPUtoGPU(h_activations,
		d_activations,
		neuronCount *
		sizeof(uint8_t));

	memcpyCPUtoGPU(h_neuronActivationCountKilling,
		d_neuronActivationCountKilling,
		neuronCount *
		sizeof(uint16_t));

	memcpyCPUtoGPU(h_forwardConnections,
		d_forwardConnections,
		connectionCount *
		sizeof(int32_t));


	memcpyCPUtoGPU(d_inputNeuronIndices,
		h_inputNeuronIndices,
		inputNeuronCount *
		sizeof(int32_t *));

	memcpyCPUtoGPU(d_inputNeuronValues,
		h_inputNeuronValues,
		inputNeuronCount *
		sizeof(uint8_t *));

	memcpyCPUtoGPU(d_outputNeuronIndices,
		h_outputNeuronIndices,
		outputNeuronCount *
		sizeof(int32_t *));

	memcpyCPUtoGPU(d_outputNeuronValues,
		h_outputNeuronValues,
		outputNeuronCount *
		sizeof(uint8_t *));
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
	fread(&inputNeuronCount, 1, sizeof(int), file);
	fread(&outputNeuronCount, 1, sizeof(int), file);

	allocateAll();

	fread(h_forwardConnections,
		connectionCount,
		sizeof(int32_t),
		file);

	fread(h_connectionWeights,
		connectionCount,
		sizeof(float),
		file);

	fread(h_activationThresholds,
		neuronCount,
		sizeof(float),
		file);

	fread(h_receivingSignal,
		connectionCount,
		sizeof(float), file);

	fread(h_excitationLevel,
		neuronCount,
		sizeof(float),
		file);

	fread(h_neuronActivationCountRebalance,
		neuronCount,
		sizeof(uint16_t),
		file);

	fread(h_activations,
		neuronCount,
		sizeof(uint8_t),
		file);

	fread(h_neuronActivationCountKilling,
		neuronCount,
		sizeof(uint16_t),
		file);


	fread(h_inputNeuronIndices,
		inputNeuronCount,
		sizeof(int32_t *),
		file);

	fread(h_inputNeuronValues,
		inputNeuronCount,
		sizeof(uint8_t *),
		file);

	fread(h_outputNeuronIndices,
		outputNeuronCount,
		sizeof(int32_t *),
		file);

	fread(h_outputNeuronValues,
		outputNeuronCount,
		sizeof(uint8_t *),
		file);


	fread(d_inputNeuronIndices,
		inputNeuronCount,
		sizeof(int32_t *),
		file);

	fread(d_inputNeuronValues,
		inputNeuronCount,
		sizeof(uint8_t *),
		file);

	fread(d_outputNeuronIndices,
		outputNeuronCount,
		sizeof(int32_t *),
		file);

	fread(d_outputNeuronValues,
		outputNeuronCount,
		sizeof(uint8_t *),
		file);

	copyToGPU();
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

int NeuralNet::getInputNeuronCount() {
	return inputNeuronCount;
}

int NeuralNet::getOutputNeuronCount() {
	return outputNeuronCount;
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

uint8_t * NeuralNet::getHostSpecialNeurons() {
	return h_specialNeurons;
}

uint16_t * NeuralNet::getHostNeuronActivationCountRebalance() {
	return h_neuronActivationCountRebalance;
}

uint16_t * NeuralNet::getHostNeuronActivationCountKilling() {
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

uint8_t * NeuralNet::getDeviceSpecialNeurons() {
	return d_specialNeurons;
}

uint16_t * NeuralNet::getDeviceNeuronActivationCountRebalance() {
	return d_neuronActivationCountRebalance;
}

uint16_t * NeuralNet::getDeviceNeuronActivationCountKilling() {
	return d_neuronActivationCountKilling;
}


uint8_t * NeuralNet::getHostInputNeuronValues() {
	return h_inputNeuronValues;
}

int32_t * NeuralNet::getHostInputNeuronIndices() {
	return h_inputNeuronIndices;
}

uint8_t * NeuralNet::getHostOutputNeuronValues() {
	return h_outputNeuronValues;
}

int32_t * NeuralNet::getHostOutputNeuronIndices() {
	return h_outputNeuronIndices;
}


uint8_t * NeuralNet::getDeviceInputNeuronValues() {
	return d_inputNeuronValues;
}

int32_t * NeuralNet::getDeviceInputNeuronIndices() {
	return d_inputNeuronIndices;
}

uint8_t * NeuralNet::getDeviceOutputNeuronValues() {
	return d_outputNeuronValues;
}

int32_t * NeuralNet::getDeviceOutputNeuronIndices() {
	return d_outputNeuronIndices;
}

