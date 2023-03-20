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

void NeuralNet::randomize() {
	randomizeNeurons(d_randState,
					 d_activationThresholds,
					 0.7,
					 1.4,
					 partitionCount,
					 neuronsPerPartition);

	createRandomConnections(d_randState,
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

	normalizeConnections(d_forwardConnections,
						 d_connectionWeights,
						 d_activationThresholds,
						 partitionCount * neuronsPerPartition,
						 maxConnectionsPerNeuron,
						 decayRate);
}

void NeuralNet::feedforward() {
	zeroizeReceivers(d_receivingSignal,
					 partitionCount,
					 neuronsPerPartition,
					 maxConnectionsPerNeuron);

	mainFeedforward(d_receivingSignal,
					 d_activations,
					 d_forwardConnections,
					 d_forwardConnectionsSub,
					 d_connectionWeights,
					 partitionCount,
					 neuronsPerPartition,
					 maxConnectionsPerNeuron);

	doNeuronReduction(d_receivingSignal,
					  partitionCount,
					  neuronsPerPartition,
					  maxConnectionsPerNeuron);
	// doExcitationDecay();
	// calculateActivations();
	// feedforwardCount++;

	// if(feedforwardCount == feedsBeforeRebalance &&
	// 		rebalanceCount == rebalancesBeforeKilling) {
	// 
	// 	determineKilledNeurons();
	// 	randomizeDeadNeurons();
	// 	ensureUniqueConnections();
	// 	zeroizeActivationCounts(d_neuronActivationCountKilling);
	// 	feedforwardCount = 0;
	// 	rebalanceCount = 0;
	// } else if(feedforwardCount == feedsBeforeRebalance) {
	//  rebalanceConnections();
	// 	normalizeConnections();
	// 	zeroizeActivationCounts(d_neuronActivationCountRebalance);
	// 	rebalanceCount++;
	// 	feedforwardCount = 0;
	// }
}
