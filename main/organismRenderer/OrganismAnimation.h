#ifndef ORGANISMANIMATION_H
#define ORGANISMANIMATION_H

#include <cstdint>

#include "Animation.h"

class Organism;

class OrganismAnimation: public Animation {
private:
	uint32_t * d_image;
	uint32_t * image;
	uint32_t blockCountGPU;

	Organism * organism;
public:
	OrganismAnimation(Organism * organism, uint32_t widthIn, uint32_t heightIn);
	void init();
	void nextFrame();
	void exit();
	void * getImage();
};

#endif