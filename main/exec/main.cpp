#include <time.h>
#include <unistd.h>

#include "renderMain.h"
#include "organism.h"
#include "NeuralNet.h"

#include "Animation.h"
#include "OrganismAnimation.h"
#include "NetworkAnimation.h"

const uint32_t WIDTH = 1920;
const uint32_t HEIGHT = 1080;
const uint32_t FRAMESPERSECOND = 144;
const uint32_t MICROSECONDSPERSECOND = 1000 * 1000;
const uint32_t NANOSECONDSPERMICROSECOND = 1000;
uint32_t microsecondsPerFrame = MICROSECONDSPERSECOND / FRAMESPERSECOND;

int main() {
	// srand(time(0));
	srand(0);

	Organism * organism = new Organism(400, 400);

	int32_t inputIndices[] = {0};
	int32_t outputIndices[] = {31};

	NeuralNet * net = new NeuralNet(
		2,
		4,
		2,
		6,
		1,
		0.8,
		0,
		1,
		0.7,
		1.4,
		1,
		1,
		0.01,
		0.01,
		1,
		1,
		inputIndices,
		outputIndices);

	net->randomize();

	uint8_t inputValues[] = {1};


	// net->printNetwork();

	net->setInputValues(inputValues);
	net->feedforward();
	net->setInputValues(inputValues);
	net->feedforward();
	net->setInputValues(inputValues);
	net->feedforward();
	net->setInputValues(inputValues);
	net->feedforward();
	net->setInputValues(inputValues);
	net->feedforward();
	net->setInputValues(inputValues);
	net->feedforward();
	net->setInputValues(inputValues);
	net->feedforward();

	net->printNetwork();

	Animation * organismAnimation = 
		new OrganismAnimation(organism, WIDTH, HEIGHT);

	Animation * networkAnimation = 
		new NetworkAnimation(net, WIDTH, HEIGHT);

	RenderMain organismRender = RenderMain(organismAnimation);
	organismRender.render();

	// RenderMain networkRender = RenderMain(networkAnimation);
	// networkRender.render();

	delete organismAnimation;
	delete networkAnimation;
	delete organism;
	delete net;
}

// int main() {
// 	srand(time(0));

// 	Organism organism = Organism(400, 400);

// 	// NeuralNet * net = new NeuralNet(
// 	// 	8,
// 	// 	15000,
// 	// 	120,
// 	// 	2500,
// 	// 	50,
// 	// 	0.8,
// 	// 	0,
// 	// 	1,
// 	// 	0.7,
// 	// 	1.4,
// 	// 	1000,
// 	// 	1000,
// 	// 	0.01,
// 	// 	0.01,
// 	// 	10,
// 	// 	10);

// 	NeuralNet * net;

// 	struct timespec start;
// 	struct timespec stop;

// 	for(int i = 1000; i <= 17100; i += 500) {
// 		printf("Allocating...\n");
// 		net = new NeuralNet(
// 			8,
// 			i,
// 			120,
// 			2500,
// 			50,
// 			0.8,
// 			0,
// 			1,
// 			0.7,
// 			1.4,
// 			1000,
// 			1000,
// 			0.01,
// 			0.01,
// 			10,
// 			10);

// 		printf("Beginning test...\n");
// 		clock_gettime(CLOCK_REALTIME, &start);
// 		net->randomize();
// 		clock_gettime(CLOCK_REALTIME, &stop);
// 		printf("Ending test...\n");
// 		uint64_t accum = ( stop.tv_sec - start.tv_sec ) * 1000000000 +
// 			( stop.tv_nsec - start.tv_nsec );

//         printf("For i=%d: Time taken: %llu.%llu seconds...\n",
//         	i,
//         	accum / 1000000000,
//         	accum % 1000000000);

//         // sleep(1);
//         printf("Freeing...\n");
//         delete net;
// 	}

// 	// RenderMain render = RenderMain(&organism);

// 	// render.render();
// }