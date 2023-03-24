#include <time.h>
#include <unistd.h>

#include "renderMain.h"
#include "organism.h"
#include "NeuralNet.h"

int main() {
	srand(time(0));

	Organism organism = Organism(400, 400);

	// NeuralNet * net = new NeuralNet(
	// 	8,
	// 	6000,
	// 	120,
	// 	2500,
	// 	50,
	// 	0.8,
	// 	0,
	// 	1,
	// 	0.7,
	// 	1.4,
	// 	1000,
	// 	0.01,
	// 	0.01,
	// 	10,
	// 	10);

	NeuralNet * net = new NeuralNet(
		1,
		4,
		1,
		2500,
		25,
		0.8,
		0,
		1,
		0.7,
		1.4,
		1000,
		1000,
		0.01,
		0.01,
		1,
		1);

	net->randomize();
	net->printNetwork();

	// RenderMain render = RenderMain(&organism);

	// render.render();

	delete net;
}