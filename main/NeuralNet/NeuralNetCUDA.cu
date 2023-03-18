__global__ void setup_kernel(curandState *state, int neurons) {

	int index = threadIdx.x + blockDim.x * blockIdx.x;
	
	if(index < neurons) {
		curand_init(1234, idx, 0, &state[idx]);
	}
}

__global__ void generate_kernel(curandState *my_curandstate, const unsigned int n, const unsigned *max_rand_int, const unsigned *min_rand_int,  unsigned int *result){

	int idx = threadIdx.x + blockDim.x*blockIdx.x;

	int count = 0;
	while (count < n){
		float myrandf = curand_uniform(my_curandstate+idx);
		myrandf *= (max_rand_int[idx] - min_rand_int[idx]+0.999999);
		myrandf += min_rand_int[idx];
		int myrand = (int)truncf(myrandf);

		assert(myrand <= max_rand_int[idx]);
		assert(myrand >= min_rand_int[idx]);
		result[myrand-min_rand_int[idx]]++;
		count++;
	}
}