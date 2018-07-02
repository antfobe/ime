#include <mpi.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
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
			f = (sin((2 * M + 1) * PI * randX) * 
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

unsigned int mc_integrationGPU(unsigned int N, int M, int k){
	unsigned int * h_hits = (unsigned int *)malloc(N * sizeof(unsigned int)); if(!h_hits) {printf("Var 'h_hits' mem alloc failed!\n"); exit(-1);}
	unsigned int * d_hits; gpuErrchk(cudaMalloc((void **)&d_hits, N * sizeof(unsigned int)));
	curandState *d_state; gpuErrchk(cudaMalloc(&d_state, sizeof(curandState)));

	//setup_kernel<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(time(NULL), d_state);
	int threadwork = 4096;
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

int main(int argc, char * argv[]) {
	if(argc != 4) {
		USAGE();
	}

	unsigned int N = (unsigned int) strtol(argv[1], (char **)NULL, 10); if(!N){ USAGE();}
	int k = (int) strtol(argv[2], (char **)NULL, 10); if(!k){ USAGE();}
	int M = (int) strtol(argv[3], (char **)NULL, 10); if(!M){ USAGE();}
	
	MPI_Init(NULL, NULL);
	/* Get the number, rank of processes */
  	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	if (world_size != 2) {
		fprintf(stderr, "World size must be two for %s\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, 1);
		exit(0);
	}	
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	//int gpu_rank = (world_rank + 1) % 2;
	
	/* gpu should bite at most 1024*1024*8 of size */
	int gpu_bite = N / (1024*1024*8);
	
	unsigned int hits = 0;
	long double result = 0.0; 
	if (world_rank != 0) {
    		MPI_Recv(&hits, 1, MPI_INT, world_rank - 1, 0,
            	MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
		hits += montecarlo_singleCPU(N / (3 * (world_size - 1)), k, M);
	} else {
    	// Set the token's (hits) value if you are process 0
		if(gpu_bite > 0){
			for (unsigned int n = gpu_bite; n < (N * 2) / 3; n += gpu_bite)
				hits += mc_integrationGPU(gpu_bite, M, k);
		} else {
			hits += mc_integrationGPU(N * 2 / 3, M, k);
		}
	}

	MPI_Send(&hits, 1, MPI_INT, (world_rank + 1) % world_size,
         0, MPI_COMM_WORLD);

	// Now process 0 can receive from the last process.
	if (world_rank == 0) {
		MPI_Recv(&hits, 1, MPI_INT, world_size - 1, 0,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		result = 2.0 * ((long double) hits/N);
		printf("Resulting integration is [%Lf], (%ld / %ld)\n", result, hits, N);
	}
	MPI_Finalize();
	return 0;
}
