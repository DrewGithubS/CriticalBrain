#include <cstdint>
#include <cstdio>

struct CPUNeuron {
	float x;
	float y;
	float z;
	uint32_t indexInGPU;
};

class NeuralNet {
private:
	int partitions;
	int neuronsPerPartition;
	int maxConnectionsPerNeuron;
	uint32_t * backwardConnections;
	uint32_t * forwardConnections;
	float * activationThreshold;
	float * receivingSignal;
	float * excitationLevel;
	uint16_t * neuronActivationCount;


	uint32_t * d_backwardConnections;
	uint32_t * d_forwardConnections;
	float * d_forwardConnectionWeights;
	float * d_activationThreshold;
	float * d_receivingSignal;
	float * d_excitationLevel;
	uint16_t * d_neuronActivationCount;
public:
	NeuralNet(int partitions, int neuronsPerPartition, int maxConnectionsPerNeuron);
	void randomize();
	void saveToFile(FILE * file);
	void loadFromFile(FILE * file);
};