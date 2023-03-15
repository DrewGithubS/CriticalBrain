#include <cstdint>

#include "GPUFunctions.h"
#include "NeuralNet.h"

NeuralNet::NeuralNet(int partitionsIn, int neuronsPerPartitionIn, int maxConnectionsPerNeuronIn) {
	partitions = partitionsIn;
	neuronsPerPartition = neuronsPerPartitionIn;
	maxConnectionsPerNeuron = maxConnectionsPerNeuronIn;

	// x
	cpuNeurons = (CPUNeuron ****) malloc(partitions * sizeof(CPUNeuron ***));
	for(int i = 0; i < partitions; i++) {
		// y
		cpuNeurons[i] = (CPUNeuron ***) malloc(partitions * sizeof(CPUNeuron **));
		for(int j = 0; j < partitions; j++) {
			// z
			cpuNeurons[i][j] = (CPUNeuron **) malloc(partitions * sizeof(CPUNeuron *));
			for(int k = 0; k < partitions; k++) {
				// neuron
				cpuNeurons[i][j][k] = (CPUNeuron *) malloc(neuronsPerPartition * sizeof(CPUNeuron));
			}
		}
	}

	int partitionCount = partitions * partitions * partitions;

	backwardConnections = (uint32_t *) malloc(partitionCount * neuronsPerPartition * maxConnectionsPerNeuron * sizeof(uint32_t));
	forwardConnections = (uint32_t *) malloc(partitionCount * neuronsPerPartition * maxConnectionsPerNeuron * sizeof(uint32_t));
	activationThreshold = (float *) malloc(partitionCount * neuronsPerPartition * sizeof(float));
	receivingSignal = (float *) malloc(partitionCount * neuronsPerPartition * maxConnectionsPerNeuron * sizeof(float));
	excitationLevel = (float *) malloc(partitionCount * neuronsPerPartition * sizeof(float));
	neuronActivationCount = (uint16_t *) malloc(partitionCount * neuronsPerPartition * sizeof(uint16_t));

	d_backwardConnections = (uint32_t *) gpuMemAlloc(partitionCount * neuronsPerPartition * maxConnectionsPerNeuron * sizeof(uint32_t));
	d_forwardConnections = (uint32_t *) gpuMemAlloc(partitionCount * neuronsPerPartition * maxConnectionsPerNeuron * sizeof(uint32_t));
	d_activationThreshold = (float *) gpuMemAlloc(partitionCount * neuronsPerPartition * sizeof(float));
	d_receivingSignal = (float *) gpuMemAlloc(partitionCount * neuronsPerPartition * maxConnectionsPerNeuron * sizeof(float));
	d_excitationLevel = (float *) gpuMemAlloc(partitionCount * neuronsPerPartition * sizeof(float));
	d_neuronActivationCount = (uint16_t *) gpuMemAlloc(partitionCount * neuronsPerPartition * sizeof(uint16_t));
}
