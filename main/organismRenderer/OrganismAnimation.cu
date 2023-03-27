#include <cstdint>
#include <iostream>

#include "OrganismAnimation.h"
#include "GPUFunctions.h"

#include "organism.h"

const uint32_t THREADSPERBLOCK = 1024;

#define LEG_WIDTH (2)
#define ORGANISM_RADIUS (10)
#define LEG_COUNT (4)
#define HAND_RADIUS (3)

#define BlockCount(x) ((x + THREADSPERBLOCK - 1)/THREADSPERBLOCK)

OrganismAnimation::OrganismAnimation(
	Organism * organismIn,
	uint32_t widthIn,
	uint32_t heightIn)
{
	organism = organismIn;
	imageWidth = widthIn;
	imageHeight = heightIn;
	init();
}

void OrganismAnimation::init()
{
	imageSize = imageWidth * imageHeight * sizeof(uint32_t);
	printf("WIDTH %d, HEIGHT: %d\n", imageWidth, imageHeight);
	image = (uint32_t *) malloc(imageSize);
	d_image = (uint32_t *) gpuMemAlloc(imageSize);
}

void OrganismAnimation::nextFrame()
{
	// cudaMemset(d_image, 0x00000000, imageSize);
	// float organismX = organism->getXPos();
	// float organismY = organism->getYPos();

	// for(int i = 0; i < LEG_COUNT; i++) {
	// 	float legAngle = organism->getLegAngle(i); // TODO: Optimize this
	// 	float legX = organism->getLegX(i);
	// 	float legY = organism->getLegY(i);

	// 	float dX = organismX - legX;
	// 	float dY = organismY - legY;

	// 	float normalizer = 1/sqrt(dX * dX + dY * dY);
	// 	float dTemp = dX;
	// 	dX = LEG_WIDTH * (-dY) * normalizer;
	// 	dY = LEG_WIDTH * dTemp * normalizer;

	// 	float legX1 = legX - dX;// * dX;
	// 	float legY1 = legY - dY;// * dY;
	// 	float legX2 = legX + dX;// * dX;
	// 	float legY2 = legY + dY;// * dY;

	// 	float orgX1 = organismX - dX;// * dX;
	// 	float orgY1 = organismY - dY;// * dY;
	// 	float orgX2 = organismX + dX;// * dX;
	// 	float orgY2 = organismY + dY;// * dY;

	// 	drawRectangle <<< blockCountGPU, THREADSPERBLOCK >>> (
	// 		d_image,
	// 		imageWidth,
	// 		imageHeight,
	// 		legX1,
	// 		legY1,
	// 		legX2,
	// 		legY2,
	// 		orgX1,
	// 		orgY1,
	// 		orgX2,
	// 		orgY2,
	// 		0xFFFF0000);

	// 	float legX3 = legX - LEG_WIDTH * dX;
	// 	float legY3 = legY + LEG_WIDTH * dY;
	// 	float legX4 = legX + LEG_WIDTH * dX;
	// 	float legY4 = legY - LEG_WIDTH * dY;

	// 	uint32_t gripStrength = 
	// 		(((organism->getGripUint(i)) << 8) | 0xFF000000);

	// 	drawCircle <<< blockCountGPU, THREADSPERBLOCK >>> (
	// 		d_image,
	// 		imageWidth,
	// 		imageHeight,
	// 		legX,
	// 		legY,
	// 		HAND_RADIUS,
	// 		gripStrength);
	// }

	// drawCircle <<< blockCountGPU, THREADSPERBLOCK >>> (
	// 	d_image,
	// 	imageWidth,
	// 	imageHeight,
	// 	organismX,
	// 	organismY,
	// 	ORGANISM_RADIUS,
	// 	0xFFFFFFFF);
}

void OrganismAnimation::exit()
{
	cudaFree(d_image);
}