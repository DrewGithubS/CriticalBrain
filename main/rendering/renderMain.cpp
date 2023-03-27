#include <time.h>
#include <unistd.h>

#include "organism.h"

#include "Animation.h"
#include "Renderer.h"
#include "renderMain.h"

const uint32_t MICROSECONDSPERSECOND = 1000 * 1000;
const uint32_t NANOSECONDSPERMICROSECOND = 1000;
uint64_t getMicrosecondsPassed(struct timespec start, struct timespec end) {
	return ((end.tv_sec - start.tv_sec) * MICROSECONDSPERSECOND) +
				((end.tv_nsec - start.tv_nsec) / NANOSECONDSPERMICROSECOND);
}

RenderMain::RenderMain(Animation * animationIn) {
	animation = animationIn;
	renderer = new Renderer(animation->getWidth(), animation->getHeight());
	printf("WIDTH %d, HEIGHT: %d\n", animation->getWidth(), animation->getHeight());
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