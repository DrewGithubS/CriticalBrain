#include <cstdint>
#include <iostream>

#include "Animation.h"
#include "GPUFunctions.h"

uint32_t Animation::getWidth()
{
	return imageWidth;
}
uint32_t Animation::getHeight()
{
	return imageHeight;
}

void * Animation::getImage()
{
	cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost);
	return image;
}