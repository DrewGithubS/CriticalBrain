__global__ void setupRand(curandState *state, int neurons) {

	int index = threadIdx.x + blockDim.x * blockIdx.x;
	
	if(index < neurons) {
		curand_init(1234, idx, 0, &state[idx]);
	}
}

__global__ void randomizeNeurons(curandState *my_curandstate, float minValue, float maxValue, int partitions, int neurons){

	int idx = threadIdx.x + blockDim.x*blockIdx.x;

	int count = 0;
	while (count < n){
		float myrandf = curand_uniform(my_curandstate+idx);
		myrandf *= (maxValue - minValue+0.999999);
		myrandf += minValue;

		result[myrand-min_rand_int[idx]]++;
	}
}