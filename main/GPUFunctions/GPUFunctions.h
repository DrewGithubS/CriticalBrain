#ifndef GPUFUNCTIONS_H
#define GPUFUNCTIONS_H

#include <cstdint>

void * gpuMemAlloc(uint32_t bytes);
void memcpyCPUtoGPU(void * to, void * from, size_t size);
void memcpyGPUtoCPU(void * to, void * from, size_t size);
void gpuMemset(void * dst, int value, size_t size);
void gpuFree(void * ptr);

__global__ void drawCircle(
	uint32_t * image,
	uint32_t imageWidth,
	uint32_t imageHeight,
	uint32_t circleX,
	uint32_t circleY,
	uint32_t circleRad,
	uint32_t pixelValue);

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
	uint32_t pixelValue);

__device__ uint32_t calcRectangleArea(
	int32_t Ax,
	int32_t Ay,
	int32_t Bx,
	int32_t By,
	int32_t Cx,
	int32_t Cy);

__device__ int32_t absoluteValue(int32_t a);

#endif