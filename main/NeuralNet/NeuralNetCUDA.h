#ifndef NEURALNETCUDA_H
#define NEURALNETCUDA_H

#include <curand.h>
#include <curand_kernel.h>

class NeuralNet;

void setupRand(
	NeuralNet * net,
	int seed);

void randomizeNeurons(
	NeuralNet * net);

void setSpecialNeurons(
	NeuralNet * net);

void createRandomConnections(
	NeuralNet * net);

void normalizeConnections(
	NeuralNet * net);

void setNetworkInputs(
	NeuralNet * net);

void mainFeedforward(
	NeuralNet * net);

void getNetworkOutputs(
	NeuralNet * net);

void doExcitationDecay(
	NeuralNet * net);

void calculateActivations(
	NeuralNet * net);

void determineKilledNeurons(
	NeuralNet * net);

void randomizeDeadNeurons(
	NeuralNet * net);

void zeroizeActivationCounts(
	uint16_t * activationCount,
	int count);

void rebalanceConnections(
	NeuralNet * net);

#endif