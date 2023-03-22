#include <cstdint>

#include "NeuralNetCUDA.h"
#include "GPUFunctions.h"
#include "NeuralNet.h"

NeuralNet::NeuralNet(
		int partitionsIn,
		int neuronsPerPartitionIn,
		int maxConnectionsPerNeuronIn,
		int feedsBeforeRebalanceIn,
		int rebalancesBeforeKillingIn)
{
	partitions = partitionsIn;
	neuronsPerPartition = neuronsPerPartitionIn;
	maxConnectionsPerNeuron = maxConnectionsPerNeuronIn;
	partitionCount = partitions * partitions * partitions;
	feedsBeforeRebalance = feedsBeforeRebalanceIn;
	feedforwardCount = 0;
	rebalancesBeforeKilling = rebalancesBeforeKillingIn;

	allocateAll();
}

void NeuralNet::allocateAll() {
	// Used for random number generation on the GPU
	d_randState = (curandState *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(curandState));

	// List of indices to postsynaptic neurons.
	d_forwardConnections = (int32_t *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(uint32_t));

	// List of indices to postsynaptic neuron receivers.
	d_forwardConnectionsSub = (int16_t *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(uint16_t));

	// Weights to use during feedforward.
	d_connectionWeights = (float *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(float));

	// Activation threshold for each neuron.
	d_activationThresholds = (float *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(float));

	// Receiving placeholders to get rid of race conditions.
	// Each neuron is responsible for summing these.
	d_receivingSignal = (float *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(float));

	// Current exitation level, when this exceeds the threshold,
	// an activation occurs. This value is set to -1 when the
	// neuron is going to be killed.
	d_excitationLevel = (float *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(float));


	d_activations = (uint8_t *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(uint8_t));


	// Incremented each time a neuron fires. Used to kill unused neurons.
	d_neuronActivationCountRebalance = (uint16_t *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(uint16_t));

	d_neuronActivationCountKilling = (uint16_t *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(uint16_t));


	h_forwardConnections = (int32_t *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(int32_t));

	h_forwardConnectionsSub = (int16_t *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(int16_t));

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


	h_tempNeuronConnectionSub = (int16_t *) gpuMemAlloc(
		partitionCount *
		neuronsPerPartition *
		sizeof(int16_t));
}

void NeuralNet::randomize()
{
	randomizeNeurons(
		d_randState,
		d_activationThresholds,
		0.7,
		1.4,
		partitionCount,
		neuronsPerPartition);

	createRandomConnections(
		d_randState,
		0,
		1,
		d_forwardConnections,
		d_forwardConnectionsSub,
		h_forwardConnections,
		h_forwardConnectionsSub,
		h_tempNeuronConnectionSub,
		d_connectionWeights,
		partitions,
		partitionCount,
		neuronsPerPartition,
		maxConnectionsPerNeuron);

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
	zeroizeReceivers(
		d_receivingSignal,
		partitionCount,
		neuronsPerPartition,
		maxConnectionsPerNeuron);

	mainFeedforward(
		d_receivingSignal,
		d_activations,
		d_forwardConnections,
		d_forwardConnectionsSub,
		d_connectionWeights,
		partitionCount,
		neuronsPerPartition,
		maxConnectionsPerNeuron);

	doNeuronReduction(
		d_receivingSignal,
		partitionCount,
		neuronsPerPartition,
		maxConnectionsPerNeuron);

	doExcitationDecay(
		d_receivingSignal,
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
			0,
			1,
			0.7,
			1.4,
			d_activationThresholds,
			d_forwardConnections,
			d_forwardConnectionsSub,
			h_forwardConnections,
			h_forwardConnectionsSub,
			h_tempNeuronConnectionSub,
			d_connectionWeights,
			d_activations,
			partitions,
			partitionCount,
			neuronsPerPartition,
			maxConnectionsPerNeuron);

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
			d_forwardConnectionsSub,
			h_forwardConnections,
			h_forwardConnectionsSub,
			h_tempNeuronConnectionSub,
			d_connectionWeights,
			d_neuronActivationCountRebalance,
			0, // TODO: put a real value here
			0.1, // TODO: Make these parts of the NN.
			0.01,
			partitions,
			partitionCount,
			neuronsPerPartition,
			maxConnectionsPerNeuron);

		normalizeConnections(
			d_forwardConnections,
			d_connectionWeights,
			d_activationThresholds,
			partitionCount * neuronsPerPartition,
			maxConnectionsPerNeuron,
			decayRate);

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

	memcpyGPUtoCPU(h_forwardConnectionsSub,
		d_forwardConnectionsSub,
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
		sizeof(int16_t));

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

	memcpyGPUtoCPU(h_receivingSignal,
		d_receivingSignal,
		partitionCount *
		neuronsPerPartition *
		maxConnectionsPerNeuron *
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

	fwrite(h_forwardConnectionsSub,
			partitionCount *
			neuronsPerPartition *
			maxConnectionsPerNeuron,
			sizeof(int16_t), file);

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

}