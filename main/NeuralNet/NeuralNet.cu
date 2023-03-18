#include <cstdint>

#include "NeuralNetCUDA.h"
#include "GPUFunctions.h"
#include "NeuralNet.h"

NeuralNet::NeuralNet(int partitionsIn, int neuronsPerPartitionIn, int maxConnectionsPerNeuronIn) {
	partitions = partitionsIn;
	neuronsPerPartition = neuronsPerPartitionIn;
	maxConnectionsPerNeuron = maxConnectionsPerNeuronIn;

	int partitionCount = partitions * partitions * partitions;

	// Is the neuron alive?
	d_isActive = (uint8_t *) gpuMemAlloc(partitionCount * neuronsPerPartition * sizeof(uint8_t));

	// List of indices to postsynaptic neurons.
	d_forwardConnections = (int32_t *) gpuMemAlloc(partitionCount * neuronsPerPartition * maxConnectionsPerNeuron * sizeof(uint32_t));

	// Weights to use during feedforward.
	d_forwardConnectionWeights = (float *) gpuMemAlloc(partitionCount * neuronsPerPartition * maxConnectionsPerNeuron * sizeof(float));

	// Activation threshold for each neuron.
	d_activationThreshold = (float *) gpuMemAlloc(partitionCount * neuronsPerPartition * sizeof(float));

	// Receiving placeholders to get rid of race conditions. Each neuron is responsible for summing these.
	d_receivingSignal = (float *) gpuMemAlloc(partitionCount * neuronsPerPartition * maxConnectionsPerNeuron * sizeof(float));

	// Current exitation level, when this exceeds the threshold, an activation occurs.
	d_excitationLevel = (float *) gpuMemAlloc(partitionCount * neuronsPerPartition * sizeof(float));

	// Incremented each time a neuron fires. Used to kill unused neurons.
	d_neuronActivationCount = (uint16_t *) gpuMemAlloc(partitionCount * neuronsPerPartition * sizeof(uint16_t));
}

void NeuralNet::randomize() {
	
}
