#ifndef ANIMATION_H
#define ANIMATION_H

#include <cstdint>

class Animation {
protected:
	uint32_t imageWidth;
	uint32_t imageHeight;

	uint32_t imageSize;
	uint32_t * d_image;
	uint32_t * image;
public:
	uint32_t getWidth();
	uint32_t getHeight();
	void * getImage();

	virtual void init() = 0;
	virtual void nextFrame() = 0;
	virtual void exit() = 0;
};

#endif