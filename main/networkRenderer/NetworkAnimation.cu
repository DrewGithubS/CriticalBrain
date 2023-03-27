#include <cstdint>
#include <iostream>

#include "NetworkAnimation.h"
#include "GPUFunctions.h"

#include "NeuralNet.h"

const uint32_t THREADSPERBLOCK = 1024;
#define BlockCount(x) ((x + THREADSPERBLOCK - 1)/THREADSPERBLOCK)

#define LEG_WIDTH (2)
#define ORGANISM_RADIUS (10)
#define LEG_COUNT (4)
#define HAND_RADIUS (3)

#define BlockCount(x) ((x + THREADSPERBLOCK - 1)/THREADSPERBLOCK)


NetworkAnimation::NetworkAnimation(
	NeuralNet * netIn,
	uint32_t widthIn,
	uint32_t heightIn)
{
	net = netIn;
	width = widthIn;
	height = heightIn;
	init();
}

void NetworkAnimation::init()
{
	imageSize = width * height * sizeof(uint32_t);
	image = (uint32_t *) malloc(imageSize);
	d_image = (uint32_t *) gpuMemAlloc(imageSize);

	d_xPositions = (float *) gpuMemAlloc(
		net->getNeuronCount() *
		sizeof(float));

	d_yPositions = (float *) gpuMemAlloc(
		net->getNeuronCount() *
		sizeof(float));
}

void NetworkAnimation::initializeAnimation() {
	float hSpacing = net->getNeuronCount();
	float vSpacing = net->getNeuronCount();

	hSpacing /= (float) sqrt((double) net->getNeuronCount());
	vSpacing /= (float) sqrt((double) net->getNeuronCount());


}

void NetworkAnimation::nextFrame()
{
	cudaMemset(d_image, 0x00000000, imageSize);

}

void NetworkAnimation::exit()
{
	cudaFree(d_image);
}