class Organism;

class RenderMain {
private:
	// struct timespec start;
	// struct timespec end;
	// uint32_t microsecondsPassed;
	Renderer * renderer;
	Animation * animation;
	Organism * organism;
public:
	RenderMain(Organism * organismIn);
	render();
}