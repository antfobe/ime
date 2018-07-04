#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <sys/time.h>
#include <curand_kernel.h>

#define PI (3.141592653589793238462643383279502884197169399375105820974)
#define POW2_32 (4294967296)
#define BLOCK_SIZE 512

#define USAGE() do { printf("Usage: ./main < N > < k > < M >\n"); exit(0); } while(0)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
__global__ void setup_kernel(unsigned int seed, curandState * state){
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(seed,	/* the seed controls the sequence of random values that are produced */
		    idx,	/* the sequence number is only important with multiple cores */
		    0,		/* the offset is how much extra we advance in the sequence for each call, can be 0 */
		    &state[idx]);	
}
__global__ void mc_integration(int seed, int threadwork, unsigned int * d_hits, unsigned int N, int M, int k){
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		curandState_t state;
		curand_init(seed, idx, 0, &state);
		double randX, f;
		d_hits[idx] = 0;
	if ( N / threadwork >= idx)
		for (int tw = 0; tw < threadwork; tw++) {
			/* pick random number between 0 and 0.5 (but not 0 itself) */
			randX = curand_uniform(&state) / 2;
			while (randX == 0.0) randX = curand_uniform(&state) / 2 ;
			/* get f(randX) */
			f = (sin((2 * abs(M) + 1) * PI * randX) * 
			     cos(2 * PI * k * randX)) / 
			     sin(PI * randX);
			/* check if f(randX) is positive (hit) */
			if (f > 0) { 
				d_hits[idx] += 1;
				__syncthreads();
			}
		}
	__syncthreads();
}

__global__ void sum_reduction(unsigned int * array, long N){
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	for(long i = 1; i < N; i <<= 1) {
		array[idx] += array[idx + i];
		__syncthreads();
	}
	__syncthreads();
}

unsigned int mc_integrationGPU(unsigned int N, int M, int k, int threadwork){
	unsigned int * h_hits = (unsigned int *)malloc(N * sizeof(unsigned int)); if(!h_hits) {printf("Var 'h_hits' mem alloc failed!\n"); exit(-1);}
	unsigned int * d_hits; gpuErrchk(cudaMalloc((void **)&d_hits, N * sizeof(unsigned int)));
	curandState *d_state; gpuErrchk(cudaMalloc(&d_state, sizeof(curandState)));

	//setup_kernel<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(time(NULL), d_state);
	mc_integration<<<(N / (BLOCK_SIZE * threadwork)) | 1 , BLOCK_SIZE>>>(time(NULL), threadwork, d_hits, N, M, k);
	gpuErrchk(cudaPeekAtLastError());

	/*gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(h_hits, d_hits, N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	unsigned int hits = 0;
	
	for(unsigned int i = 0; i < N; i++){
		if(h_hits[i]) hits+=h_hits[i];
	}*/

	sum_reduction<<<(N / BLOCK_SIZE) | 1, BLOCK_SIZE>>>(d_hits, N);
	gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaDeviceSynchronize());
	
	unsigned int r_hits = 0;
	gpuErrchk(cudaMemcpy(&r_hits, d_hits, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	//printf("r_hits = (%d)\n", r_hits);	

	gpuErrchk(cudaFree(d_hits));
	gpuErrchk(cudaFree(d_state));

	free(h_hits);
	return r_hits;
}

unsigned int montecarlo_singleCPU(unsigned int N, unsigned int k, unsigned int M){
	double randX, f;
	unsigned int r_hits = 0;
	srand(time(NULL));

	for (unsigned int i = 0; i < N; i++) {
		randX = ((double)rand()/(double)(RAND_MAX)) * .5;
		while (randX == 0.0) randX = ((double)rand()/(double)(RAND_MAX)) * .5;
		f = (sin((2 * M + 1) * PI * randX) * 
		     cos(2 * PI * k * randX)) / 
		     sin(PI * randX);
		if (f > 0) r_hits++;
	}

	return r_hits;
}

double montecarlo_OMP(unsigned int N, unsigned int k, unsigned int M, double * f2){
	double randX, f = 0.0, f_hits = 0.0;
	unsigned int i =0;
	//unsigned int r_hits = 0, i =0;
	srand(time(NULL));

	//#pragma omp parallel shared(r_hits) private(i, randX, f)
	#pragma omp parallel shared(f_hits) private(i, randX, f)
	{
	#pragma omp for schedule (static)
	for (i = 0; i < N; i++) {
		randX = ((double)rand()/(double)(RAND_MAX)) * .5;
		while (randX == 0.0) randX = ((double)rand()/(double)(RAND_MAX)) * .5;
		f = (sin((2 * M + 1) * PI * randX) * 
		     cos(2 * PI * k * randX)) / 
		     sin(PI * randX);
		f2[0] += (f * f);
		f_hits += f;
		//if (f > 0) r_hits++;
	}
	}
	return f_hits;
}  
int tvsub(struct timeval *result, struct timeval *t2, struct timeval *t1){
	long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
	result->tv_sec = diff / 1000000;
	result->tv_usec = diff % 1000000;

	return (diff<0);
}

int main(int argc, char * argv[]) {
	if(argc != 4) {
		USAGE();
	}

	unsigned int N = (unsigned int) strtol(argv[1], (char **)NULL, 10); if(!N){ USAGE();}
	int k = (int) strtol(argv[2], (char **)NULL, 10); if(!k){ USAGE();}
	int M = (int) strtol(argv[3], (char **)NULL, 10); if(!M){ USAGE();}
	struct timeval begin, end;
	
	/* gpu should bite at most 1024*1024*8 of size */
	int gpu_bite = N / (1024*1024*4);
	
	unsigned int hits = 0;
	long double result = 0.0, err = 0.0;
	double f2[1]; f2[0] =0.0; 
	
	gettimeofday(&begin, NULL);
	//hits += mc_integrationGPU(N, M, k, 2048);
	result += montecarlo_OMP(N, M, k, f2);
	//hits += montecarlo_singleCPU(N, M, k);
	gettimeofday(&end, NULL);

	//result = 2.0 * ((long double) hits/N);
	err = sqrt((f2[0]/N - result * result) / N);
	result = (long double) result / N;
	if (M < 0) {result *= -1;}
	tvsub(&end, &end, &begin);
	char wot[256]; sprintf(wot, "%ld.%06ld", end.tv_sec, end.tv_usec);
	double timu; sscanf(wot, "%lf", &timu);
	//printf("Resulting integration is [%Lf], in (%ld.%06ld)s, (%ld / %ld)\n", result, end.tv_sec, end.tv_usec, hits, N);
	//printf("Resulting integration is [%Lf] - err [%lf], in (%f)s\n", result * result * N, sqrt((f2[0]/N)), timu);
	printf("Resulting integration is [%Lf] - err [%lf], in (%f)s\n", result, err, timu);
	return 0;
}
