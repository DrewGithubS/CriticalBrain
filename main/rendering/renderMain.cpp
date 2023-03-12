#include <time.h>
#include <unistd.h>

#include "../organism/organism.h"

#include "Animation.h"
#include "Renderer.h"

const uint32_t WIDTH = 1920;
const uint32_t HEIGHT = 1080;
const uint32_t FRAMESPERSECOND = 144;
const uint32_t MICROSECONDSPERSECOND = 1000 * 1000;
const uint32_t NANOSECONDSPERMICROSECOND = 1000;
uint32_t microsecondsPerFrame = MICROSECONDSPERSECOND / FRAMESPERSECOND;

uint64_t getMicrosecondsPassed(struct timespec start, struct timespec end) {
	return ((end.tv_sec - start.tv_sec) * MICROSECONDSPERSECOND) + ((end.tv_nsec - start.tv_nsec) / NANOSECONDSPERMICROSECOND);
}

void RenderMain::RenderMain(Organism * organismIn) {
	renderer = new Renderer(WIDTH, HEIGHT);
	animation = new Animation(WIDTH, HEIGHT);
	organism = organismIn;
}

void RenderMain::render() {
	// clock_gettime(CLOCK_REALTIME, &start);
	// usleep(microsecondsPerFrame);

	while(renderer->getActiveStatus()) {
		// clock_gettime(CLOCK_REALTIME, &end);
		// microsecondsPassed = getMicrosecondsPassed(start, end);
		// if(microsecondsPassed >= microsecondsPerFrame) {
			renderer->checkForEvent();
			// renderer->getFrame();
			animation->nextFrame();
			renderer->setFrame(animation->getImage());
			renderer->render();
		// } else {
			// usleep(microsecondsPerFrame - microsecondsPassed);
		// }
	}
}