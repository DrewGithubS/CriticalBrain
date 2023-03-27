#include <iostream>

#include "GPUFunctions.h"

void * gpuMemAlloc(uint32_t bytes) {
	void * output;
	cudaError_t err = cudaMalloc(&output, bytes);
	if ( err != cudaSuccess ) {
		std::cout << cudaGetErrorString(err) << std::endl;
		return NULL;
	}

	return output;
};

void memcpyCPUtoGPU(void * to, void * from, size_t size) {
	cudaMemcpy(to, from, size, cudaMemcpyHostToDevice);
}

void memcpyGPUtoCPU(void * to, void * from, size_t size) {
	cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost);
}

void gpuMemset(void * dst, int value, size_t size) {
	cudaMemset(dst, value, size);
}

void gpuFree(void * ptr) {
	cudaFree(ptr);
}

__device__ int32_t absoluteValue(int32_t a)
{
	return a < 0 ? -a : a;
}

__device__ uint32_t calcRectangleArea(
	int32_t Ax,
	int32_t Ay,
	int32_t Bx,
	int32_t By,
	int32_t Cx,
	int32_t Cy)
{
	return absoluteValue( 
				(Bx * Ay - Ax * By) +
				(Cx * By - Bx * Cy) +
				(Ax * Cy - Cx * Ay));

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
	uint32_t pixelValue)
{
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

__global__ void drawCircle(
	uint32_t * image,
	uint32_t imageWidth,
	uint32_t imageHeight,
	uint32_t circleX,
	uint32_t circleY,
	uint32_t circleRad,
	uint32_t pixelValue)
{
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