#include <cstdint>

#include "NeuralNetCUDA.h"
#include "GPUFunctions.h"
#include "NeuralNet.h"

NeuralNet::NeuralNet(int partitionsIn,
					 int neuronsPerPartitionIn,
					 int maxConnectionsPerNeuronIn,
					 int feedsBeforeRebalanceIn,
					 int rebalancesBeforeKillingIn) {

	partitions = partitionsIn;
	neuronsPerPartition = neuronsPerPartitionIn;
	maxConnectionsPerNeuron = maxConnectionsPerNeuronIn;
	partitionCount = partitions * partitions * partitions;
	feedsBeforeRebalance = feedsBeforeRebalanceIn;
	feedforwardCount = 0;
	rebalancesBeforeKilling = rebalancesBeforeKillingIn;

	// Used for random number generation on the GPU
	d_randState = (curandState *) gpuMemAlloc(partitionCount * neuronsPerPartition * sizeof(curandState));

	// If the value is -1, the neuron is dead.
	// Used to indicate with partition a neuron is in.
	d_partitionLoc = (int16_t *) gpuMemAlloc(partitionCount * sizeof(uint16_t));

	// List of indices to postsynaptic neurons.
	d_forwardConnections = (int32_t *) gpuMemAlloc(partitionCount * neuronsPerPartition * maxConnectionsPerNeuron * sizeof(uint32_t));

	// Weights to use during feedforward.
	d_forwardConnectionWeights = (float *) gpuMemAlloc(partitionCount * neuronsPerPartition * maxConnectionsPerNeuron * sizeof(float));

	// Activation threshold for each neuron.
	d_activationThresholds = (float *) gpuMemAlloc(partitionCount * neuronsPerPartition * sizeof(float));

	// Receiving placeholders to get rid of race conditions. Each neuron is responsible for summing these.
	d_receivingSignal = (float *) gpuMemAlloc(partitionCount * neuronsPerPartition * maxConnectionsPerNeuron * sizeof(float));

	// Current exitation level, when this exceeds the threshold, an activation occurs.
	d_excitationLevel = (float *) gpuMemAlloc(partitionCount * neuronsPerPartition * sizeof(float));

	// Incremented each time a neuron fires. Used to kill unused neurons.
	d_neuronActivationCountRebalance = (uint16_t *) gpuMemAlloc(partitionCount * neuronsPerPartition * sizeof(uint16_t));
	d_neuronActivationCountKilling = (uint16_t *) gpuMemAlloc(partitionCount * neuronsPerPartition * sizeof(uint16_t));
}

void NeuralNet::randomize() {
	randomizeNeurons(d_randState,
					 d_activationThresholds,
					 d_partitionLoc,
					 0.7,
					 1.4,
					 partitionCount,
					 neuronsPerPartition);

	createRandomConnections(d_randState,
							d_partitionLoc,
							d_forwardConnections,
							d_forwardConnectionWeights,
							partitionCount,
							neuronsPerPartition,
							maxConnectionsPerNeuron);

	// ensureUniqueConnections();
}

void NeuralNet::feedforward() {
	// zeroizeReceivers();
	// mainFeedforward();
	// doNeuronReduction();
	// calculateActivations();
	// doExcitationDecay();
	// feedforwardCount++;

	// if(feedforwardCount == feedsBeforeRebalance && rebalanceCount == rebalancesBeforeKilling) {
	// 	determineKilledNeurons();
	// 	randomizeDeadNeurons();
	// 	ensureUniqueConnections();
	// 	zeroizeActivationCounts(d_neuronActivationCountKilling);
	// 	feedforwardCount = 0;
	// 	rebalanceCount = 0;
	// } else if(feedforwardCount == feedsBeforeRebalance) {
	//  rebalanceConnections();
	// 	zeroizeActivationCounts(d_neuronActivationCountRebalance);
	// 	rebalanceCount++;
	// 	feedforwardCount = 0;
	// }
}
