#include <cstdint>
#include <iostream>

#include "Animation.h"
#include "GPUFunctions.h"

#include "organism.h"

const uint32_t THREADSPERBLOCK = 1024;

#define LEG_WIDTH (2)
#define ORGANISM_RADIUS (10)
#define LEG_COUNT (4)
#define HAND_RADIUS (3)

#define BlockCount(x) ((x + THREADSPERBLOCK - 1)/THREADSPERBLOCK)

__device__ int32_t absoluteValue(int32_t a) {
	return a < 0 ? -a : a;
}

__device__ uint32_t calcRectangleArea(
	int32_t Ax,
	int32_t Ay,
	int32_t Bx,
	int32_t By,
	int32_t Cx,
	int32_t Cy) {



	return absoluteValue( 
				(Bx * Ay - Ax * By) +
				(Cx * By - Bx * Cy) +
				(Ax * Cy - Cx * Ay));

}

__global__ void drawCircle(
	uint32_t * image,
	uint32_t imageWidth,
	uint32_t imageHeight,
	uint32_t circleX,
	uint32_t circleY,
	uint32_t circleRad,
	uint32_t pixelValue) {

	uint32_t index = blockIdx.x *blockDim.x + threadIdx.x;

	if(index < imageWidth * imageHeight) {
		uint32_t pixelX = index % imageWidth;
		uint32_t pixelY = index / imageWidth;

		int32_t distX = circleX - pixelX;
		int32_t distY = circleY - pixelY;
		distX *= distX;
		distY *= distY;
		int32_t maxDist = circleRad * circleRad;

		if((distX + distY) < maxDist) {
			image[index] = pixelValue;
		}
	}
}

// Receives four sets of coordinates
// It determines if the point is in the rectangle
// If it is, it sets the pixel value to the provided input value
__global__ void drawRectangle(
	uint32_t * image,
	uint32_t imageWidth,
	uint32_t imageHeight,
	uint32_t Ax,
	uint32_t Ay,
	uint32_t Bx,
	uint32_t By,
	uint32_t Cx,
	uint32_t Cy,
	uint32_t Dx,
	uint32_t Dy,
	uint32_t pixelValue) {

	uint32_t index = blockIdx.x *blockDim.x + threadIdx.x;

	if(index < imageWidth * imageHeight) {
		uint32_t pixelX = index % imageWidth;
		uint32_t pixelY = index / imageWidth;

		// Somehow summing the area of these four triangles can determine if a
		// pixel is in the rectangle. If the area of these four triangles is
		// less than the area of the rectangle, the point is in the rectangle.
		uint32_t trianglesA = 
					calcRectangleArea(Ax, Ay, pixelX, pixelY, Dx, Dy);
				trianglesA +=
					calcRectangleArea(Dx, Dy, pixelX, pixelY, Cx, Cy);
				trianglesA +=
					calcRectangleArea(Cx, Cy, pixelX, pixelY, Bx, By);
				trianglesA +=
					calcRectangleArea(pixelX, pixelY, Bx, By, Ax, Ay);

		// To get the area of the triangle instead of the rectangle,
		// Divide by 2.
		trianglesA /= 2;

		// Rectangles can be defined by only three poinnts
		uint32_t rectA = calcRectangleArea(Ax, Ay, Bx, By, Cx, Cy);

		if(trianglesA <= rectA) {
			// The point is in the rectangle
			image[index] = pixelValue;
		}
	}
}

Animation::Animation(uint32_t widthIn, uint32_t heightIn) {
	width = widthIn;
	height = heightIn;
	init();
}

void Animation::init() {
	imageSize = width * height * sizeof(uint32_t);
	image = (uint32_t *) malloc(imageSize);
	d_image = (uint32_t *) gpuMemAlloc(imageSize);
	blockCountGPU = BlockCount(imageWidth * imageHeight);
}

void Animation::nextFrame(Organism * organism) {
	cudaMemset(d_image, 0x00000000, imageSize);
	float organismX = organism->getXPos();
	float organismY = organism->getYPos();

	for(int i = 0; i < LEG_COUNT; i++) {
		float legAngle = organism->getLegAngle(i); // TODO: Optimize this
		float legX = organism->getLegX(i);
		float legY = organism->getLegY(i);

		float dX = organismX - legX;
		float dY = organismY - legY;

		float normalizer = 1/sqrt(dX * dX + dY * dY);
		float dTemp = dX;
		dX = LEG_WIDTH * (-dY) * normalizer;
		dY = LEG_WIDTH * dTemp * normalizer;

		float legX1 = legX - dX;// * dX;
		float legY1 = legY - dY;// * dY;
		float legX2 = legX + dX;// * dX;
		float legY2 = legY + dY;// * dY;

		float orgX1 = organismX - dX;// * dX;
		float orgY1 = organismY - dY;// * dY;
		float orgX2 = organismX + dX;// * dX;
		float orgY2 = organismY + dY;// * dY;

		drawRectangle <<< blockCountGPU, THREADSPERBLOCK >>> (
			d_image,
			width,
			height,
			legX1,
			legY1,
			legX2,
			legY2,
			orgX1,
			orgY1,
			orgX2,
			orgY2,
			0xFFFF0000);

		float legX3 = legX - LEG_WIDTH * dX;
		float legY3 = legY + LEG_WIDTH * dY;
		float legX4 = legX + LEG_WIDTH * dX;
		float legY4 = legY - LEG_WIDTH * dY;

		uint32_t gripStrength = 
			(((organism->getGripUint(i)) << 8) | 0xFF000000);

		drawCircle <<< blockCountGPU, THREADSPERBLOCK >>> (
			d_image,
			width,
			height,
			legX,
			legY,
			HAND_RADIUS,
			gripStrength);
	}

	drawCircle <<< blockCountGPU, THREADSPERBLOCK >>> (
		d_image,
		width,
		height,
		organismX,
		organismY,
		ORGANISM_RADIUS,
		0xFFFFFFFF);
}

void Animation::exit() {
	cudaFree(d_image);
}

void * Animation::getImage() {
	cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost);
	return image;
}