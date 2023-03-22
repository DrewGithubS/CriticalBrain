// NOTE: Positive activations for good things and negative for bad things.
// The goal of the neural net should be to maximize perception.



#include <cstdint>

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
	int16_t * d_forwardConnectionsSub,
	int32_t * h_forwardConnections,
	int16_t * h_forwardConnectionsSub,
	int16_t * h_tempNeuronConnectionSub,
	int partitions,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron);



__global__ void d_setupRand(
	curandState * state,
	int neurons)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	
	if(index < neurons) {
		curand_init(1234, index, 0, &state[index]);
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
		activationThresholds[index] *= (maxValue - minValue + 0.999999);
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
	    weight *= (maxWeight - minWeight + 0.999999);
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

__global__ void d_zeroizeReceivers(
	float * receivingSignal,
	int connections)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < connections) {
		receivingSignal[index] = 0;
	}
}

__global__ void d_feedForward(
	float * receivingSignal,
	uint8_t * activations,
	int32_t * forwardConnections,
	int16_t * forwardConnectionsSub,
	float * connectionWeights,
	int connections,
	int connectionsPerNeuron,
	int neurons)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < connections) {
		int neuron = index / connectionsPerNeuron;
		receivingSignal[
			forwardConnections[index] *
			connectionsPerNeuron +
			forwardConnectionsSub[index]] = 
				activations[neuron] ? connectionWeights[index] : 0;
	}
}

/*
26 connections, 0-25.
00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
00+13 01+14 02+15 03+16 04+17 05+18 06+19 07+20 08+21 09+22 10+23 11+24 12+25

13 connections, 0-12
00    01    02    03    04    05    06    07    08    09    10    11    12
00 01 02 03 04 05 06 07 08 09 10 11 12
00+06 01+07 02+08 03+09 04+10 05+11 06+12

7 connections, 0-6
00    01    02    03    04    05    06
00 01 02 03 04 05 06s
00+03 01+04 02+05 03+06

4 connections, 0-3
00    01    02    03
00 01 02 03
00+02 01+03

2 connections 0-1
00    01
00 01
00+01

1 connections 0-0.
00
*/
// Increment is (connections / 2)
// Max is less than (connections / 2)



__global__ void d_doNeuronReduction(
	float * receivingSignal,
	int connections,
	int connectionsPerNeuron)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < connections) {
		int neuron = index / connectionsPerNeuron;
		int connectionIndex = index % connectionsPerNeuron;
		int tempConnections = connectionsPerNeuron;

		while(tempConnections > 0) {
			int increment = tempConnections / 2;
			if(connectionIndex < increment) {

				receivingSignal[neuron *
					connectionsPerNeuron +
					connectionIndex] += 
						receivingSignal[neuron *
						connectionsPerNeuron +
						connectionIndex + increment];

			}

			tempConnections = (tempConnections + 1) / 2;
		}
	}
}

__global__ void d_doExcitationDecay(
	float * receivingSignal,
	float * excitationLevel,
	float decayRate,
	int neurons,
	int connectionsPerNeuron)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < neurons) {
		int receptionIndex = index * connectionsPerNeuron;

		excitationLevel[index] = receivingSignal[receptionIndex] * decayRate;
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

void randomizeNeurons(
	curandState * curandStates,
	float * activationThresholds,
	float minValue,
	float maxValue,
	int16_t partitions,
	int neuronsPerPartition)
{
	int neurons = partitions * neuronsPerPartition;

	d_setupRand <<< BlockCount(neurons), THREADSPERBLOCK >>> (
		curandStates,
		neurons);

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
	int16_t * d_forwardConnectionsSub,
	int32_t * h_forwardConnections,
	int16_t * h_forwardConnectionsSub,
	int16_t * h_tempNeuronConnectionSub,
	float * connectionWeights,
	int partitions,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron)
{
	srand(time(0));

	int neurons = partitionCount * neuronsPerPartition;
	int connections = neurons * connectionsPerNeuron;

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

	ensureUniqueConnections(
		d_forwardConnections,
		d_forwardConnectionsSub,
		h_forwardConnections,
		h_forwardConnectionsSub,
		h_tempNeuronConnectionSub,
		partitions,
		partitionCount,
		neuronsPerPartition,
		connectionsPerNeuron);
	
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

void zeroizeReceivers(
	float * receivingSignal,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron)
{
	int neurons = partitionCount * neuronsPerPartition;
	int connections = neurons * connectionsPerNeuron;

	d_zeroizeReceivers <<< 
		BlockCount(connections),
		THREADSPERBLOCK >>> (
			receivingSignal,
			connections);
}

void mainFeedforward(
	float * receivingSignal,
	uint8_t * activations,
	int32_t * forwardConnections,
	int16_t * forwardConnectionsSub,
	float * connectionWeights,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron)
{
	int neurons = partitionCount * neuronsPerPartition;
	int connections = neurons * connectionsPerNeuron;

	d_feedForward <<< 
		BlockCount(connections),
		THREADSPERBLOCK >>> (
			receivingSignal,
			activations,
			forwardConnections,
			forwardConnectionsSub,
			connectionWeights,
			connections,
			connectionsPerNeuron,
			neurons);
}

void doNeuronReduction(
	float * receivingSignal,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron)
{
	int neurons = partitionCount * neuronsPerPartition;
	int connections = neurons * connectionsPerNeuron;

	d_doNeuronReduction <<< 
		BlockCount(connections),
		THREADSPERBLOCK >>> (
			receivingSignal,
			connections,
			connectionsPerNeuron);

}

void doExcitationDecay(
	float * receivingSignal,
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
			receivingSignal,
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
	int16_t * d_forwardConnectionsSub,
	int32_t * h_forwardConnections,
	int16_t * h_forwardConnectionsSub,
	int16_t * h_tempNeuronConnectionSub,
	float * connectionWeights,
	uint8_t * activations,
	int partitions,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron)
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
		d_forwardConnectionsSub,
		h_forwardConnections,
		h_forwardConnectionsSub,
		h_tempNeuronConnectionSub,
		partitions,
		partitionCount,
		neuronsPerPartition,
		connectionsPerNeuron);
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
	int16_t * d_forwardConnectionsSub,
	int32_t * h_forwardConnections,
	int16_t * h_forwardConnectionsSub,
	int16_t * h_tempNeuronConnectionSub,
	float * connectionWeights,
	uint16_t * activationCount,
	uint16_t minimumActivations,
	float changeConstant,
	float minimumWeightValue,
	int partitions,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron)
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
		d_forwardConnectionsSub,
		h_forwardConnections,
		h_forwardConnectionsSub,
		h_tempNeuronConnectionSub,
		partitions,
		partitionCount,
		neuronsPerPartition,
		connectionsPerNeuron);
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
    x_f *= (maxValueX - minValueX + 0.999999);
    x_f += minValueX;
    int dx = (int) truncf(x_f);

    float y_f = curand_uniform(state + index);
    y_f *= (maxValueY - minValueY + 0.999999);
    y_f += maxValueY;
    int dy = (int) truncf(y_f);

    float z_f = curand_uniform(state + index);
    z_f *= (maxValueZ - minValueZ + 0.999999);
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
    z_f *= (neuronsPerPartition - 1 + 0.999999);
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
	int16_t * d_forwardConnectionsSub,
	int32_t * h_forwardConnections,
	int16_t * h_forwardConnectionsSub,
	int16_t * h_tempNeuronConnectionSub,
	int partitions,
	int partitionCount,
	int neuronsPerPartition,
	int connectionsPerNeuron)
{
	int neurons = partitionCount * neuronsPerPartition;
	int connections = neurons * connectionsPerNeuron;

	memcpyGPUtoCPU(h_forwardConnections, 
				   d_forwardConnections,
				   connections * sizeof(int32_t));

	// This is probably very slow. I need a way to optimize this
	// to work in parallel or work faster.
	for(int i = 0; i < neurons; i++) {
		for(int j = 0; j < connectionsPerNeuron; j++) {
			while(h_forwardConnections[i * connectionsPerNeuron + j] == -1 ||
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

	memset(h_tempNeuronConnectionSub,
		   0,
		   partitions * neuronsPerPartition * sizeof(int16_t));

	// Complicated way to assign each forward connection a unique index
	// I wish this could be parallelized.
	for(int i = 0; i < connections; i++) {
		h_forwardConnectionsSub[i] = 
			h_tempNeuronConnectionSub[h_forwardConnections[i]];

		h_tempNeuronConnectionSub[h_forwardConnections[i]]++;
	}

	memcpyCPUtoGPU(d_forwardConnections, 
				   h_forwardConnections,
				   connections * sizeof(int32_t));

	memcpyCPUtoGPU(d_forwardConnectionsSub, 
				   h_forwardConnectionsSub,
				   connections * sizeof(int16_t));
}