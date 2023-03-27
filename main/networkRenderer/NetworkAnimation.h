#ifndef NETWORKANIMATION_H
#define NETWORKANIMATION_H

#include <cstdint>

#include "Animation.h"

class NeuralNet;

class NetworkAnimation: public Animation {
private:
	uint32_t width;
	uint32_t height;

	uint32_t imageWidth;
	uint32_t imageHeight;
	
	float * d_xPositions;
	float * d_yPositions;

	uint32_t imageSize;
	uint32_t * d_image;
	uint32_t * image;

	NeuralNet * net;
public:
	NetworkAnimation(NeuralNet * net, uint32_t widthIn, uint32_t heightIn);
	void initializeAnimation();
	void init();
	void nextFrame();
	void exit();
	void * getImage();
};

#endif