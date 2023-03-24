#ifndef NEURALNETCUDA_H
#define NEURALNETCUDA_H

#include <curand.h>
#include <curand_kernel.h>

void setupRand(
	NeuralNet * net);

void randomizeNeurons(
	NeuralNet * net);

void createRandomConnections(
	NeuralNet * net);

void normalizeConnections(
	NeuralNet * net);

void mainFeedforward(
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