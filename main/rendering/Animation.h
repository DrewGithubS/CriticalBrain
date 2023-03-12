#ifndef ANIMATION_H
#define ANIMATION_H

#include <cstdint>

class Animation {
private:
	uint32_t width;
	uint32_t height;

	uint32_t imageWidth;
	uint32_t imageHeight;
	
	uint32_t * data;
	uint32_t * d_data;
	uint32_t * d_compressableImage;

	uint32_t ** d_proximityData;

	uint32_t imageSize;
	uint32_t * d_image;
	uint32_t * image;
	uint32_t blockCountGPU;
public:
	Animation(uint32_t widthIn, uint32_t heightIn);
	void init();
	void nextFrame();
	void exit();
	void * getImage();
};

#endif