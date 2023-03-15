#include "renderMain.h"
#include "organism.h"

int main() {
	Organism organism = Organism(400, 400);

	RenderMain render = RenderMain(&organism);

	render.render();
}