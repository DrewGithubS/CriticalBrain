// NOTE: Positive activations for good things and negative for bad things.
// The goal of the neural net should be to maximize perception.



#include <cstdint>
#include <cstdio>

#include "NeuralNetCUDA.h"
#include "GPUFunctions.h"

const uint32_t THREADSPERBLOCK = 1024;	
#define BlockCount(x) ((x + THREADSPERBLOCK - 1)/THREADSPERBLOCK)

uint8_t arrayContains(int32_t * arr, int32_t item, int len);

__device__ int d_genRandomNeuron(
	curandState * state,
	int index,
	int connectionsPerNeuron,
	int neuronsPerPartition,
	int partitions);

int h_genRandomNeuron(
	int index,
	int connectionsPerNeuron,
	int neuronsPerPartition,
	int partitions);

void ensureUniqueConnections(
	int32_t * d_forwardConnections,
	int32_t * h_forwardConnections,
	int partitions,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron,
	int inputNeurons);



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

	if(index < connectionCount) {
		forwardConnections[index] = d_genRandomNeuron(
			curandStates,
			index,
			connectionsPerNeuron,
			neuronsPerPartition,
			partitions);

		float weight = curand_uniform(curandStates + index);
	    weight *= (maxWeight - minWeight);
	    weight += minWeight;
		
	    connectionWeights[index] = weight;
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

__global__ void d_feedForward(
	float * excitationLevel,
	uint8_t * activations,
	int32_t * forwardConnections,
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
		
		if(neuron < (neurons - outputNeurons)) {
			atomicAdd(&excitationLevel[
				forwardConnections[index]],
				activations[neuron] ? connectionWeights[index] : 0);
		}
	}
}

__global__ void d_doExcitationDecay(
	float * excitationLevel,
	float decayRate,
	int neurons,
	int connectionsPerNeuron)
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
	uint16_t minimumActivations,
	int neurons)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < neurons) {
		activations[index] = 
			activationCount[index] < minimumActivations ? 255 : 0;
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
	curandState * curandStates,
	int seed,
	int16_t partitions,
	int neuronsPerPartition) {

	int neurons = partitions * neuronsPerPartition;

	d_setupRand <<< BlockCount(neurons), THREADSPERBLOCK >>> (
		curandStates,
		neurons,
		seed);
}

void randomizeNeurons(
	curandState * curandStates,
	float * activationThresholds,
	float minValue,
	float maxValue,
	int16_t partitions,
	int neuronsPerPartition)
{
	int neurons = partitions * neuronsPerPartition;

	d_randomizeNeurons <<< BlockCount(neurons), THREADSPERBLOCK >>> (
		curandStates,
		activationThresholds,
		minValue,
		maxValue,
		partitions,
		neurons);
}

void createRandomConnections(
	curandState * curandStates,
	float minWeight,
	float maxWeight,
	int32_t * d_forwardConnections,
	int32_t * h_forwardConnections,
	float * connectionWeights,
	int partitions,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron,
	int inputNeurons)
{
	int neurons = partitionCount * neuronsPerPartition;
	int connections = neurons * connectionsPerNeuron;


	printf("Calling GPU function...\n"); fflush(stdout);
	d_createRandomConnections <<< 
		BlockCount(connections),
		THREADSPERBLOCK >>> (
			curandStates,
			minWeight,
			maxWeight,
			d_forwardConnections,
			connectionWeights,
			partitions,
			neuronsPerPartition,
			neurons,
			connectionsPerNeuron,
			connections);

	printf("Ensuring unique connections...\n");
	ensureUniqueConnections(
		d_forwardConnections,
		h_forwardConnections,
		partitions,
		partitionCount,
		neuronsPerPartition,
		connectionsPerNeuron,
		inputNeurons);
	
}

void normalizeConnections(
	int32_t * forwardConnections,
	float * connectionWeights,
	float * activationThresholds,
	int neurons,
	int connectionsPerNeuron,
	float decayRate)
{
	d_normalizeNeurons <<< 
		BlockCount(neurons),
		THREADSPERBLOCK >>> (
			forwardConnections,
			connectionWeights,
			activationThresholds,
			neurons,
			connectionsPerNeuron,
			decayRate);
}

void mainFeedforward(
	float * excitationLevel,
	uint8_t * activations,
	int32_t * forwardConnections,
	float * connectionWeights,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron,
	int outputNeurons)
{
	int neurons = partitionCount * neuronsPerPartition;
	int connections = neurons * connectionsPerNeuron;

	d_feedForward <<< 
		BlockCount(connections),
		THREADSPERBLOCK >>> (
			excitationLevel,
			activations,
			forwardConnections,
			connectionWeights,
			connections,
			connectionsPerNeuron,
			neurons,
			outputNeurons);
}

void doExcitationDecay(
	float * excitationLevel,
	float decayRate,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron)
{
	int neurons = partitionCount * neuronsPerPartition;
	int connections = neurons * connectionsPerNeuron;

	d_doExcitationDecay <<< 
		BlockCount(connections),
		THREADSPERBLOCK >>> (
			excitationLevel,
			decayRate,
			neurons,
			connectionsPerNeuron);

}

void calculateActivations(
	float * excitationLevel,
	float * activationThresholds,
	uint8_t * activations,
	uint16_t * activationCount1,
	uint16_t * activationCount2,
	int partitionCount,
	int neuronsPerPartition)
{
	int neurons = partitionCount * neuronsPerPartition;

	d_calculateActivations <<< 
		BlockCount(neurons),
		THREADSPERBLOCK >>> (
			excitationLevel,
			activationThresholds,
			activations,
			activationCount1,
			activationCount2,
			neurons);
}

void determineKilledNeurons(
	uint16_t * activationCount,
	uint8_t * activations,
	uint16_t minimumActivations,
	int partitionCount,
	int neuronsPerPartition)
{
	int neurons = partitionCount * neuronsPerPartition;

	d_determineKilledNeurons <<< 
		BlockCount(neurons),
		THREADSPERBLOCK >>> (
			activationCount,
			activations,
			minimumActivations,
			neurons);
}

void randomizeDeadNeurons(
	curandState * curandStates,
	float minWeight,
	float maxWeight,
	float minActivation,
	float maxActivation,
	float * activationThresholds,
	int32_t * d_forwardConnections,
	int32_t * h_forwardConnections,
	float * connectionWeights,
	uint8_t * activations,
	int partitions,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron,
	int inputNeurons)
{
	int neurons = partitionCount * neuronsPerPartition;
	int connections = neurons * connectionsPerNeuron;

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

	ensureUniqueConnections(
		d_forwardConnections,
		h_forwardConnections,
		partitions,
		partitionCount,
		neuronsPerPartition,
		connectionsPerNeuron,
		inputNeurons);
}

void zeroizeActivationCounts(
	uint16_t * activationCount,
	int partitionCount,
	int neuronsPerPartition)
{
	int neurons = partitionCount * neuronsPerPartition;

	d_zeroizeActivationCounts <<< 
		BlockCount(neurons),
		THREADSPERBLOCK >>> (
			activationCount,
			neurons);
}

void rebalanceConnections(
	int32_t * d_forwardConnections,
	int32_t * h_forwardConnections,
	float * connectionWeights,
	uint16_t * activationCount,
	uint16_t minimumActivations,
	float changeConstant,
	float minimumWeightValue,
	int partitions,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron,
	int inputNeurons)
{
	int neurons = partitionCount * neuronsPerPartition;

	d_rebalanceConnections <<< 
		BlockCount(neurons),
		THREADSPERBLOCK >>> (
			d_forwardConnections,
			connectionWeights,
			activationCount,
			minimumActivations,
			minimumWeightValue,
			changeConstant,
			neurons,
			connectionsPerNeuron);

	ensureUniqueConnections(
		d_forwardConnections,
		h_forwardConnections,
		partitions,
		partitionCount,
		neuronsPerPartition,
		connectionsPerNeuron,
		inputNeurons);
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
	int neuronIndex = index / connectionsPerNeuron;
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
    x_f *= (maxValueX - minValueX);
    x_f += minValueX;
    int dx = (int) truncf(x_f);

    float y_f = curand_uniform(state + index);
    y_f *= (maxValueY - minValueY);
    y_f += maxValueY;
    int dy = (int) truncf(y_f);

    float z_f = curand_uniform(state + index);
    z_f *= (maxValueZ - minValueZ);
    z_f += maxValueZ;
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
    z_f *= (neuronsPerPartition - 1);
    z_f += 0;
    newNeuronIndex = neuronPartitionIndex + (int) truncf(z_f);
    // Guarantees the connection isn't pointing at itself.
    newNeuronIndex += (newNeuronIndex >= neuronIndex);

    return newNeuronIndex;
}

int h_genRandomNeuron(
	int index,
	int connectionsPerNeuron,
	int neuronsPerPartition,
	int partitions)
{
	int neuronIndex = index / connectionsPerNeuron;
	int partitionIndex = neuronIndex / neuronsPerPartition;
	int partitionX = (partitionIndex % partitions); 
	int partitionY = (partitionIndex / partitions) % partitions;
	int partitionZ = (partitionIndex / (partitions * partitions));
	
	int minValueX = partitionX == 0 ? 0 : -1;
	int minValueY = partitionY == 0 ? 0 : -1;
	int minValueZ = partitionZ == 0 ? 0 : -1;

	int maxValueX = partitionX == (partitions - 1) ? 0 : 1;
	int maxValueY = partitionY == (partitions - 1) ? 0 : 1;
	int maxValueZ = partitionZ == (partitions - 1) ? 0 : 1;

    int dx = (rand() % (maxValueX - minValueX + 1)) + minValueX;
    int dy = (rand() % (maxValueY - minValueY + 1)) + minValueY;
    int dz = (rand() % (maxValueZ - minValueZ + 1)) + minValueZ;

    int neuronPartitionIndex = 
    		(partitionZ + dz) * partitions * partitions +
			(partitionY + dy) * partitions + 
			(partitionX + dx);

    neuronPartitionIndex *= neuronsPerPartition;

	int newNeuronIndex = neuronPartitionIndex;

    newNeuronIndex += rand() % (neuronsPerPartition);

    // Guarantees the connection isn't pointing at itself.
    newNeuronIndex += (newNeuronIndex >= neuronIndex);

    return newNeuronIndex;
}

uint8_t arrayContains(int32_t * arr, int32_t item, int len)
{
	for(int i = 0; i < len; i++) {
		if(arr[i] == item) {
			return 1;
		}
	}

	return 0;
}

// I need to optimize this somehow.
void ensureUniqueConnections(
	int32_t * d_forwardConnections,
	int32_t * h_forwardConnections,
	int partitions,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron,
	int inputNeurons)
{
	int neurons = partitionCount * neuronsPerPartition;
	int connections = neurons * connectionsPerNeuron;

	printf("Copying to CPU...\n"); fflush(stdout);
	memcpyGPUtoCPU(h_forwardConnections, 
				   d_forwardConnections,
				   connections * sizeof(int32_t));

	// This is probably very slow. I need a way to optimize this
	// to work in parallel or work faster.
	for(int i = 0; i < neurons; i++) {
		printf("Randomizing for %d neuron...\n", i); fflush(stdout);
		for(int j = 0; j < connectionsPerNeuron; j++) {
			printf("Randomizing for %d connection...\n", j); fflush(stdout);
			while(h_forwardConnections[i * connectionsPerNeuron + j] < inputNeurons ||
					h_forwardConnections[i * connectionsPerNeuron + j] == -1 ||
					arrayContains(
						&h_forwardConnections[i * connectionsPerNeuron],
						h_forwardConnections[i * connectionsPerNeuron + j],
						j)) {

				h_forwardConnections[i * connectionsPerNeuron + j] =
					h_genRandomNeuron(i,
									  connectionsPerNeuron,
									  neuronsPerPartition,
									  partitions);
				
			}
		}
	}

	memcpyCPUtoGPU(d_forwardConnections, 
				   h_forwardConnections,
				   connections * sizeof(int32_t));
}