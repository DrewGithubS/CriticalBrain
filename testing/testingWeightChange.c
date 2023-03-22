#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
	const int n = 120;

	srand(time(0));

	int count[n];

	int sum = 0;

	float deltas[n];

	for(int i = 0; i < n; i++) {
		count[i] = rand() % 250;
		sum += count[i];
		printf("%d\n", count[i]);
	}

	float sumOfDeltas = 0;

	printf("\n");

	for(int i = 0; i < n; i++) {
		deltas[i] = n * ( (float) (count[i]) - ((float) sum / (float) n) )/ ((float) sum);
		sumOfDeltas += deltas[i];
		printf("%f\n", deltas[i]);
	}

	printf("\n%f\n", sumOfDeltas);
}