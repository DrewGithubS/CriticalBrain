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
	// 	15000,
	// 	120,
	// 	2500,
	// 	50,
	// 	0.8,
	// 	0,
	// 	1,
	// 	0.7,
	// 	1.4,
	// 	1000,
	// 	1000,
	// 	0.01,
	// 	0.01,
	// 	10,
	// 	10);

	// NeuralNet * net = new NeuralNet(
	// 	2,
	// 	4,
	// 	2,
	// 	2500,
	// 	25,
	// 	0.8,
	// 	0,
	// 	1,
	// 	0.7,
	// 	1.4,
	// 	1000,
	// 	1000,
	// 	0.01,
	// 	0.01,
	// 	1,
	// 	1);

	NeuralNet * net;

	struct timespec start;
	struct timespec stop;

	for(int i = 1000; i <= 17100; i += 500) {
		printf("Allocating...\n");
		net = new NeuralNet(
			8,
			i,
			120,
			2500,
			50,
			0.8,
			0,
			1,
			0.7,
			1.4,
			1000,
			1000,
			0.01,
			0.01,
			10,
			10);

		printf("Beginning test...\n");
		clock_gettime(CLOCK_REALTIME, &start);
		net->randomize();
		clock_gettime(CLOCK_REALTIME, &stop);
		printf("Ending test...\n");
		uint64_t accum = ( stop.tv_sec - start.tv_sec ) * 1000000000 +
			( stop.tv_nsec - start.tv_nsec );

        printf("For i=%d: Time taken: %llu.%llu seconds...\n",
        	i,
        	accum / 1000000000,
        	accum % 1000000000);

        sleep(1);
        printf("Freeing...\n");
        delete net;
	}
	// net->printNetwork();

	// RenderMain render = RenderMain(&organism);

	// render.render();

	// delete net;
}