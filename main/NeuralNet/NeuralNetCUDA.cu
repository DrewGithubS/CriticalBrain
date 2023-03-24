// NOTE: Positive activations for good things and negative for bad things.
// The goal of the neural net should be to maximize perception.

enum {
	NORMAL_NEURON,
	OUTPUT_NEURON,
	INPUT_NEURON,
} NeuronTypes;

#include <cstdint>
#include <cstdio>

#include "NeuralNetCUDA.h"
#include "NeuralNet.h"
#include "GPUFunctions.h"

const uint32_t THREADSPERBLOCK = 1024;	
#define BlockCount(x) ((x + THREADSPERBLOCK - 1)/THREADSPERBLOCK)

__device__ uint8_t arrayContains(int32_t * arr, int32_t item, int len);

__device__ int d_genRandomNeuron(
	curandState * state,
	int index,
	int connectionsPerNeuron,
	int neuronsPerPartition,
	int partitions);

__global__ void d_setupRand(
	curandState * state,
	int neurons,
	int seed)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	
	if(index < neurons) {
		curand_init(seed, index, 0, &state[index]);
	}
}

__global__ void d_randomizeNeurons(
	curandState * curandStates,
	float * activationThresholds,
	float minValue,
	float maxValue,
	int partitions,
	int neurons)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < neurons) {
		activationThresholds[index] = curand_uniform(curandStates + index);
		activationThresholds[index] *= (maxValue - minValue);
		activationThresholds[index] += minValue;
	}
}

__global__ void d_createRandomConnections(
	curandState * curandStates,
	float minWeight,
	float maxWeight,
	int32_t * forwardConnections,
	float * connectionWeights,
	int partitions,
	int neuronsPerPartition,
	int neuronCount,
	int connectionsPerNeuron,
	int connectionCount)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < neuronCount) {
		int cIndex = index * connectionsPerNeuron;
		for(int i = 0; i < connectionsPerNeuron; i++) {
			forwardConnections[cIndex + i] = d_genRandomNeuron(
				curandStates,
				index,
				connectionsPerNeuron,
				neuronsPerPartition,
				partitions);

			float weight = curand_uniform(curandStates + index);
		    weight *= (maxWeight - minWeight);
		    weight += minWeight;
			
		    connectionWeights[cIndex + i] = weight;
		}
	}
}

__global__ void d_ensureUniqueConnections(
	curandState * curandStates,
	int32_t * forwardConnections,
	uint8_t * specialNeurons,
	int partitions,
	int inputNeurons,
	int connectionsPerNeuron,
	int neuronsPerPartition,
	int neuronCount)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < neuronCount) {
		for(int j = 0; j < connectionsPerNeuron; j++) {
			while(
				forwardConnections[index * connectionsPerNeuron + j]
					== index ||
				forwardConnections[index * connectionsPerNeuron + j]
					< inputNeurons ||
				forwardConnections[index * connectionsPerNeuron + j]
					== -1 ||
				specialNeurons[
					forwardConnections[index * connectionsPerNeuron + j]]
						== INPUT_NEURON ||
				arrayContains(
					&forwardConnections[index * connectionsPerNeuron],
					forwardConnections[index * connectionsPerNeuron + j],
					j)) {

				forwardConnections[index * connectionsPerNeuron + j] =
					d_genRandomNeuron(curandStates,
						index,
						connectionsPerNeuron,
						neuronsPerPartition,
						partitions);
				
			}
		}
	}
}

__global__ void d_normalizeNeurons(
	int32_t * forwardConnections,
	float * connectionWeights,
	float * activationThresholds,
	int neurons,
	int connectionsPerNeuron,
	float decayRate)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < neurons) {
		float sumTotal = 0;
		int connectionBegin = index * connectionsPerNeuron;
		int neuronIndex;
		for(int i = 0; i < connectionsPerNeuron; i++) {
			neuronIndex = connectionBegin + i;
			sumTotal += connectionWeights[neuronIndex] / 
				activationThresholds[forwardConnections[neuronIndex]];
		}

		// Ensures the sum of all the connection rates is
		// 1 after excitation decay.
		float divider = 1 / (sumTotal * decayRate);

		for(int i = 0; i < connectionsPerNeuron; i++) {
			neuronIndex = connectionBegin + i;
			connectionWeights[neuronIndex] *= divider;
		}
	}
}

__global__ void d_setNetworkInputs(
	uint8_t * inputValues,
	uint8_t * activations,
	int32_t * inputNeuronIndices,
	int inputNeuronCount)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < inputNeuronCount) {
		activations[inputNeuronIndices[index]] = inputValues[index];
	}
}

__global__ void d_feedForward(
	float * excitationLevel,
	uint8_t * activations,
	int32_t * forwardConnections,
	uint8_t * specialNeurons,
	float * connectionWeights,
	int connections,
	int connectionsPerNeuron,
	int neurons,
	int outputNeurons)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < connections) {
		int neuron = index / connectionsPerNeuron;
		if(index % connectionsPerNeuron == 0) {
			excitationLevel[neuron] = 0;
		}
		
		if(specialNeurons[index] != OUTPUT_NEURON) {
			atomicAdd(&excitationLevel[
				forwardConnections[index]],
				activations[neuron] ? connectionWeights[index] : 0);
		}
	}
}

__global__ void d_getNetworkOutputs(
	uint8_t * outputValues,
	uint8_t * activations,
	int32_t * outputNeuronIndices,
	int outputNeuronCount)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < outputNeuronCount) {
		outputValues[index] = activations[outputNeuronIndices[index]];
	}
}

__global__ void d_doExcitationDecay(
	float * excitationLevel,
	float decayRate,
	int neurons)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < neurons) {
		excitationLevel[index] *= decayRate;
	}
}

__global__ void d_calculateActivations(
	float * excitationLevel,
	float * activationThresholds,
	uint8_t * activations,
	uint16_t * activationCount1,
	uint16_t * activationCount2,
	int neurons)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < neurons) {
		activations[index] =
			excitationLevel[index] > activationThresholds[index];

		excitationLevel[index] = 
			excitationLevel[index] > activationThresholds[index] ? 
				0 : excitationLevel[index];

		activationCount1[index] +=
			excitationLevel[index] > activationThresholds[index];

		activationCount2[index] +=
			excitationLevel[index] > activationThresholds[index];
	}
}

__global__ void d_determineKilledNeurons(
	uint16_t * activationCount,
	uint8_t * activations,
	uint8_t * specialNeurons,
	uint16_t minimumActivations,
	int neurons)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < neurons) {
		activations[index] =
			specialNeurons[index] == NORMAL_NEURON *
			(activationCount[index] < minimumActivations ? 255 : 0);
	}
}

__global__ void d_randomizeDeadNeurons(
	curandState * curandStates,
	float minWeight,
	float maxWeight,
	float minActivation,
	float maxActivation,
	float * activationThresholds,
	int32_t * forwardConnections,
	float * connectionWeights,
	uint8_t * activations,
	int partitions,
	int neuronsPerPartition,
	int neuronCount,
	int connectionsPerNeuron,
	int connectionCount)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < connectionCount) {
		int connectionIndex = index % connectionsPerNeuron;
		int neuronIndex = index / connectionsPerNeuron;

		// Randomize new neuron activation thresholds
		if(connectionIndex == 0) {
			activationThresholds[neuronIndex] =
				curand_uniform(curandStates + index);

			activationThresholds[neuronIndex] *=
				(maxActivation - minActivation + 0.999999);

			activationThresholds[neuronIndex] +=
				minActivation;
		}

		// Randomize forward connections
		if(activations[neuronIndex] == 255) {
			forwardConnections[index] = d_genRandomNeuron(
				curandStates,
				index,
				connectionsPerNeuron,
				neuronsPerPartition,
				partitions);

			float weight = curand_uniform(curandStates + index);
		    weight *= (maxWeight - minWeight + 0.999999);
		    weight += minWeight;
			
		    connectionWeights[index] = weight;
		}

		// Randomize connections from neurons that died.
		if(activations[forwardConnections[index]] == 255) {
			forwardConnections[index] = d_genRandomNeuron(
				curandStates,
				index,
				connectionsPerNeuron,
				neuronsPerPartition,
				partitions);
		}
	}
}

__global__ void d_zeroizeActivationCounts(
	uint16_t * activationCount,
	int neurons)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < neurons) {
		activationCount[index] = 0;
	}
}

__global__ void d_rebalanceConnections(
	int32_t * forwardConnections,
	float * connectionWeights,
	uint16_t * activationCount,
	uint16_t minimumActivations,
	float minimumWeightValue,
	float changeConstant,
	int neurons,
	int connectionsPerNeuron)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < neurons) {
		float sum = 0;
		int connectionBegin = index * connectionsPerNeuron;
		int neuronIndex;
		for(int i = 0; i < connectionsPerNeuron; i++) {
			neuronIndex = forwardConnections[connectionBegin + i];
			sum += (float) activationCount[neuronIndex];
		}

		for(int i = 0; i < connectionsPerNeuron; i++) {
			connectionWeights[connectionBegin + i] += 
				connectionsPerNeuron * 
				( (float) (activationCount[i]) - 
					(sum / (float) connectionsPerNeuron) )
						/ (sum);
		}

		for(int i = 0; i < connectionsPerNeuron; i++) {
			if(
			activationCount[forwardConnections[connectionBegin + i]] <
				minimumActivations || 
			fabs(connectionWeights[connectionBegin + i]) <
				minimumWeightValue) {

				forwardConnections[connectionBegin + i] = -1;
			}
		}
	}
}

void setupRand(
	NeuralNet * net,
	int seed)
{

	curandState * curandStates = net->getDeviceRandState();
	int neurons = net->getNeuronCount();

	d_setupRand <<< BlockCount(neurons), THREADSPERBLOCK >>> (
		curandStates,
		neurons,
		seed);

	cudaDeviceSynchronize();
}

void randomizeNeurons(
	NeuralNet * net)
{
	curandState * curandStates = net->getDeviceRandState();
	float * activationThresholds = net->getDeviceActivationThresholds();
	float minActivation = net->getMinActivationValue();
	float maxActivation = net->getMaxActivationValue();
	int16_t partitions = net->getPartitions();
	int neurons = net->getNeuronCount();

	d_randomizeNeurons <<< BlockCount(neurons), THREADSPERBLOCK >>> (
		curandStates,
		activationThresholds,
		minActivation,
		maxActivation,
		partitions,
		neurons);

	cudaDeviceSynchronize();
}

void createRandomConnections(
	NeuralNet * net)
{

	curandState * curandStates = net->getDeviceRandState();
	float minWeight = net->getMinWeightValue();
	float maxWeight = net->getMaxWeightValue();
	int32_t * d_forwardConnections = net->getDeviceForwardConnections();
	uint8_t * specialNeurons = net->getDeviceSpecialNeurons();
	float * d_connectionWeights = net->getDeviceConnectionWeights();
	int partitions = net->getPartitions();
	int partitionCount = net->getPartitionCount();
	int neuronsPerPartition = net->getNeuronsPerPartition();
	int neurons = net->getNeuronCount();
	int connectionsPerNeuron = net->getMaxConnectionsPerNeuron();
	int connections = net->getConnectionCount();
	int inputNeurons = net->getInputNeuronCount();


	d_createRandomConnections <<< 
		BlockCount(connections),
		THREADSPERBLOCK >>> (
			curandStates,
			minWeight,
			maxWeight,
			d_forwardConnections,
			d_connectionWeights,
			partitions,
			neuronsPerPartition,
			neurons,
			connectionsPerNeuron,
			connections);

	d_ensureUniqueConnections <<< 
		BlockCount(neurons),
		THREADSPERBLOCK >>> (
			curandStates,
			d_forwardConnections,
			specialNeurons,
			partitions,
			inputNeurons,
			connectionsPerNeuron,
			neuronsPerPartition,
			neurons);

	cudaDeviceSynchronize();
}

void normalizeConnections(
	NeuralNet * net)
{
	int32_t * forwardConnections = net->getDeviceForwardConnections();
	float * connectionWeights = net->getDeviceConnectionWeights();
	float * activationThresholds = net->getDeviceActivationThresholds();
	int neurons = net->getNeuronCount();
	int connectionsPerNeuron = net->getMaxConnectionsPerNeuron();
	float decayRate = net->getDecayRate();

	d_normalizeNeurons <<< 
		BlockCount(neurons),
		THREADSPERBLOCK >>> (
			forwardConnections,
			connectionWeights,
			activationThresholds,
			neurons,
			connectionsPerNeuron,
			decayRate);

	cudaDeviceSynchronize();
}

void setNetworkInputs(
	NeuralNet * net)
{
	uint8_t * h_inputValues = net->getHostInputNeuronValues();
	uint8_t * activations = net->getDeviceActivations();
	uint8_t * d_inputValues = net->getDeviceInputNeuronValues();
	int32_t * d_inputNeuronIndices = net->getDeviceInputNeuronIndices();
	uint32_t inputNeuronCount = net->getInputNeuronCount();

	memcpyCPUtoGPU(
		d_inputValues,
		h_inputValues,
		inputNeuronCount * sizeof(uint8_t));

	d_setNetworkInputs <<< 
		BlockCount(inputNeuronCount),
		THREADSPERBLOCK >>> (
			d_inputValues,
			activations,
			d_inputNeuronIndices,
			inputNeuronCount);
}

void mainFeedforward(
	NeuralNet * net)
{
	float * excitationLevel = net->getDeviceExcitationLevel();
	uint8_t * activations = net->getDeviceActivations();
	int32_t * forwardConnections = net->getDeviceForwardConnections();
	uint8_t * specialNeurons = net->getDeviceSpecialNeurons();
	float * connectionWeights = net->getDeviceConnectionWeights();
	int connections = net->getConnectionCount();
	int neurons = net->getNeuronCount();
	int connectionsPerNeuron = net->getMaxConnectionsPerNeuron();
	int outputNeurons = net->getOutputNeuronCount();

	d_feedForward <<< 
		BlockCount(connections),
		THREADSPERBLOCK >>> (
			excitationLevel,
			activations,
			forwardConnections,
			specialNeurons,
			connectionWeights,
			connections,
			connectionsPerNeuron,
			neurons,
			outputNeurons);

	cudaDeviceSynchronize();
}

void getNetworkOutputs(
	NeuralNet * net)
{
	uint8_t * h_outputValues = net->getHostOutputNeuronValues();
	uint8_t * activations = net->getDeviceActivations();
	uint8_t * d_outputValues = net->getDeviceOutputNeuronValues();
	int32_t * d_outputNeuronIndices = net->getDeviceOutputNeuronIndices();
	uint32_t outputNeuronCount = net->getOutputNeuronCount();

	

	d_getNetworkOutputs <<< 
		BlockCount(outputNeuronCount),
		THREADSPERBLOCK >>> (
			h_outputValues,
			activations,
			d_outputNeuronIndices,
			outputNeuronCount);

	memcpyGPUtoCPU(
		h_outputValues,
		d_outputValues,
		outputNeuronCount * sizeof(uint8_t));
}

void doExcitationDecay(
	NeuralNet * net)
{
	float * excitationLevel = net->getDeviceExcitationLevel();
	float decayRate = net->getDecayRate();
	int neurons = net->getNeuronCount();

	d_doExcitationDecay <<< 
		BlockCount(neurons),
		THREADSPERBLOCK >>> (
			excitationLevel,
			decayRate,
			neurons);

	cudaDeviceSynchronize();
}

void calculateActivations(
	NeuralNet * net)
{
	float * excitationLevel = net->getDeviceExcitationLevel();
	float * activationThresholds = net->getDeviceActivationThresholds();
	uint8_t * activations = net->getDeviceActivations();
	uint16_t * activationCount1 = net->getHostNeuronActivationCountRebalance();
	uint16_t * activationCount2 = net->getHostNeuronActivationCountKilling();
	int neurons = net->getNeuronCount();

	d_calculateActivations <<< 
		BlockCount(neurons),
		THREADSPERBLOCK >>> (
			excitationLevel,
			activationThresholds,
			activations,
			activationCount1,
			activationCount2,
			neurons);

	cudaDeviceSynchronize();
}

void determineKilledNeurons(
	NeuralNet * net)
{
	uint16_t * activationCount = net->getHostNeuronActivationCountKilling();
	uint8_t * activations = net->getDeviceActivations();
	uint8_t * specialNeurons = net->getDeviceSpecialNeurons();
	uint16_t minimumKillingActivations = net->getMinimumKillingActivations();
	int neurons = net->getNeuronCount();

	d_determineKilledNeurons <<< 
		BlockCount(neurons),
		THREADSPERBLOCK >>> (
			activationCount,
			activations,
			specialNeurons,
			minimumKillingActivations,
			neurons);

	cudaDeviceSynchronize();
}

void randomizeDeadNeurons(
	NeuralNet * net)
{
	curandState * curandStates = net->getDeviceRandState();
	float minWeight = net->getMinWeightValue();
	float maxWeight = net->getMaxWeightValue();
	float minActivation = net->getMinActivationValue();
	float maxActivation = net->getMaxActivationValue();
	float * activationThresholds = net->getDeviceActivationThresholds();
	int32_t * d_forwardConnections = net->getDeviceForwardConnections();
	float * connectionWeights = net->getDeviceConnectionWeights();
	uint8_t * activations = net->getDeviceActivations();
	uint8_t * specialNeurons = net->getDeviceSpecialNeurons();
	int partitions = net->getPartitions();
	int partitionCount = net->getPartitionCount();
	int neuronsPerPartition = net->getNeuronsPerPartition();
	int neurons = net->getNeuronCount();
	int connectionsPerNeuron = net->getMaxConnectionsPerNeuron();
	int connections = net->getConnectionCount();
	int inputNeurons = net->getInputNeuronCount();

	d_randomizeDeadNeurons <<< 
		BlockCount(neurons),
		THREADSPERBLOCK >>> (
			curandStates,
			minWeight,
			maxWeight,
			minActivation,
			maxActivation,
			activationThresholds,
			d_forwardConnections,
			connectionWeights,
			activations,
			partitions,
			neuronsPerPartition,
			neurons,
			connectionsPerNeuron,
			connections);

	d_ensureUniqueConnections <<< 
		BlockCount(neurons),
		THREADSPERBLOCK >>> (
			curandStates,
			d_forwardConnections,
			specialNeurons,
			partitions,
			inputNeurons,
			connectionsPerNeuron,
			neuronsPerPartition,
			neurons);

	cudaDeviceSynchronize();
}

void zeroizeActivationCounts(
	uint16_t * activationCount,
	int count)
{
	d_zeroizeActivationCounts <<< 
		BlockCount(count),
		THREADSPERBLOCK >>> (
			activationCount,
			count);

	cudaDeviceSynchronize();
}

void rebalanceConnections(
	NeuralNet * net)
{
	curandState * curandStates = net->getDeviceRandState();
	int32_t * d_forwardConnections = net->getDeviceForwardConnections();
	int32_t * h_forwardConnections = net->getHostForwardConnections();
	float * connectionWeights = net->getDeviceConnectionWeights();
	uint16_t * activationCount = net->getHostNeuronActivationCountRebalance();
	uint16_t minimumRebalanceActivations =
		net->getMinimumRebalanceActivations();
	float changeConstant = net->getChangeConstant();
	float weightKillValue = net->getWeightKillValue();
	uint8_t * specialNeurons = net->getDeviceSpecialNeurons();
	int partitions = net->getPartitions();
	int partitionCount = net->getPartitionCount();
	int neuronsPerPartition = net->getNeuronsPerPartition();
	int connectionsPerNeuron = net->getMaxConnectionsPerNeuron();
	int inputNeurons = net->getInputNeuronCount();
	int neurons = partitionCount * neuronsPerPartition;

	d_rebalanceConnections <<< 
		BlockCount(neurons),
		THREADSPERBLOCK >>> (
			d_forwardConnections,
			connectionWeights,
			activationCount,
			minimumRebalanceActivations,
			weightKillValue,
			changeConstant,
			neurons,
			connectionsPerNeuron);

	d_ensureUniqueConnections <<< 
		BlockCount(neurons),
		THREADSPERBLOCK >>> (
			curandStates,
			d_forwardConnections,
			specialNeurons,
			partitions,
			inputNeurons,
			connectionsPerNeuron,
			neuronsPerPartition,
			neurons);

	cudaDeviceSynchronize();
}



/*****************************************************************************/

/*
          zyx
0  1  2   000 001 002
3  4  5   010 011 012
6  7  8   020 021 022

9  10 11  100 101 102
12 13 14  110 111 112
15 16 17  120 121 122

18 19 20  200 201 202
21 22 23  210 211 212
24 25 26  220 221 222

x = (n % 3)
y = ((n / 3) % 3)
z = (n / (3 * 3))
*/
__device__ int d_genRandomNeuron(
	curandState * state,
	int index,
	int connectionsPerNeuron,
	int neuronsPerPartition,
	int partitions)
{
	int neuronIndex = index;
	int partitionIndex = neuronIndex / neuronsPerPartition;
	int partitionX = (partitionIndex % partitions); 
	int partitionY = (partitionIndex / partitions) % partitions;
	int partitionZ = (partitionIndex / (partitions * partitions));
	
	float minValueX = partitionX == 0 ? 0 : -1;
	float minValueY = partitionY == 0 ? 0 : -1;
	float minValueZ = partitionZ == 0 ? 0 : -1;

	float maxValueX = partitionX == (partitions - 1) ? 0 : 1;
	float maxValueY = partitionY == (partitions - 1) ? 0 : 1;
	float maxValueZ = partitionZ == (partitions - 1) ? 0 : 1;

	

	float x_f = curand_uniform(state + index);
    x_f *= (maxValueX - minValueX + 0.999999);
    x_f += minValueX;
    int dx = (int) truncf(x_f);

    float y_f = curand_uniform(state + index);
    y_f *= (maxValueY - minValueY + 0.999999);
    y_f += minValueY;
    int dy = (int) truncf(y_f);

    float z_f = curand_uniform(state + index);
    z_f *= (maxValueZ - minValueZ + 0.999999);
    z_f += minValueZ;
    int dz = (int) truncf(z_f);

    int neuronPartitionIndex = 
    		(partitionZ + dz) * partitions * partitions +
			(partitionY + dy) * partitions + 
			(partitionX + dx);

    neuronPartitionIndex *= neuronsPerPartition;

	int newNeuronIndex;
	z_f = curand_uniform(state + index);
	// Subtracting one here so that I can guarantee this connection
	// Doesn't point at itself
    z_f *= (neuronsPerPartition - 1 + 0.999999);
    z_f += 0;
    newNeuronIndex = neuronPartitionIndex + (int) truncf(z_f);
    // Guarantees the connection isn't pointing at itself.
    newNeuronIndex += (newNeuronIndex >= neuronIndex);

    return newNeuronIndex;
}

__device__ uint8_t arrayContains(int32_t * arr, int32_t item, int len)
{
	for(int i = 0; i < len; i++) {
		if(arr[i] == item) {
			return 1;
		}
	}

	return 0;
}