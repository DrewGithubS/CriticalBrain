#include "Animation.h"
class Renderer;

class RenderMain {
private:
	// struct timespec start;
	// struct timespec end;
	// uint32_t microsecondsPassed;
	Renderer * renderer;
	Animation * animation;
public:
	RenderMain(Animation * animation);
	void render();
};