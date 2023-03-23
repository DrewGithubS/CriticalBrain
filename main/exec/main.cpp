#include "renderMain.h"
#include "organism.h"
#include "NeuralNet.h"

int main() {
	Organism organism = Organism(400, 400);

	NeuralNet * net = new NeuralNet(
		8,
		6000,
		120,
		2500,
		25,
		0.8,
		0,
		1,
		0.7,
		1.4,
		1000,
		0.01,
		0.01,
		10,
		10);

	RenderMain render = RenderMain(&organism);

	render.render();

	delete net;
}