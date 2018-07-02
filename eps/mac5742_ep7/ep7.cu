/* Adapted from code at http://www.speedup.ch/workshops/w43_2014/tutorial/html/cuda_mpi_1.html */
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

#define PI (3.141592653589793238462643383279502884197169399375105820974)
#define USAGE() do { printf("Usage: ./main < N > < M > < k >\n"); exit(0); } while(0)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
/* kernel function on device */
__global__ void montecarlo_GPU(int seed, unsigned int * d_hits, unsigned int N, unsigned int M, unsigned int k){
	extern __shared__ unsigned int sdata[];
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	curandState_t state;
	curand_init(seed, idx, 0, &state);
	/* pick random number between 0 and 0.5 (but not 0 itself) */
	double randX = curand_uniform(&state) / 2;
	while (randX == 0.0) randX = curand_uniform(&state) / 2 ;
	/* get f(randX) */
	double f = (sin((2 * M + 1) * PI * randX) * 
		     cos(2 * PI * k * randX)) / 
		     sin(PI * randX);
	if (f>0) sdata[threadIdx.x] = 1;
	else 	 sdata[threadIdx.x] = 0;

	__syncthreads();

	for (int s=blockDim.x >> 1; s>0; s>>=1) {
		if (threadIdx.x < s) {
			sdata[threadIdx.x] += sdata[threadIdx.x + s];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0)  {d_hits[blockIdx.x] = sdata[0]; printf("sdata0 = %d\n", sdata[0]);}
}


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void add_1_block(unsigned int *sum, unsigned int N){
	if (N>blockDim.x) return;
	extern __shared__ unsigned int sdata[];
	unsigned int tid = threadIdx.x;
	if (tid<N) sdata[tid]=sum[tid];
	else       sdata[tid]=0.0;

	__syncthreads();
	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		 }
	__syncthreads();
	}
	if (tid == 0)  sum[0] = sdata[0];
}


/* Host function. Function output calculated with device vectors */
unsigned int dmc_GPU(unsigned int * d_hits, unsigned int N, unsigned int M, unsigned int k){
	cudaDeviceProp prop;
	cudaGetDeviceProperties (&prop, 0);

	unsigned int blocksize = 4 * 64,
		  gridsize = prop.multiProcessorCount,
		  sBytes   = blocksize * sizeof(unsigned int);
	dim3 dimBlock(blocksize);
	dim3 dimGrid(gridsize);

	unsigned int hits;  /* device array storing the partial sums */
	cudaMalloc((void **) &d_hits, blocksize * sizeof(unsigned int)); /* temp. memory on device */

	/* call the kernel function with  dimGrid.x * dimBlock.x threads */
	montecarlo_GPU <<< dimGrid, dimBlock, sBytes>>>(time(NULL), d_hits, N, M, k);
	/* power of 2 as number of treads per block */
	const unsigned int oneBlock = 2 << (int)ceil(log(dimGrid.x + 0.0) / log(2.0));
	add_1_block <<< 1, oneBlock, oneBlock *sizeof(unsigned int)>>>(d_hits, dimGrid.x);
	
	cudaMemcpy(&hits, d_hits, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	
	//printf("hits = %d\n", hits);

	cudaFree(d_hits);
	return hits;
}

unsigned int montecarlo_singleCPU(unsigned int N, unsigned int k, unsigned int M){
	double randX, f;
	unsigned int hits = 0;
	srand(time(NULL));

	for (unsigned int i = 0; i < N; i++) {
		randX = ((double)rand()/(double)(RAND_MAX)) * .5;
		while (randX == 0.0) randX = ((double)rand()/(double)(RAND_MAX)) * .5;
		f = (sin((2 * M + 1) * PI * randX) * 
		     cos(2 * PI * k * randX)) / 
		     sin(PI * randX);
		if (f > 0) hits++;
	}

	return hits;
}

/* main function on host */
int main(int argc, char * argv[]) {
	if(argc != 4) {
		USAGE();
	}

	long N = (long) strtol(argv[1], (char **)NULL, 10); if(!N){ USAGE();}
	int M = (int) strtol(argv[2], (char **)NULL, 10); if(!M){ USAGE();}
	int k = (int) strtol(argv[3], (char **)NULL, 10); if(!k){ USAGE();}
	MPI_Init(NULL, NULL);
	unsigned int *h_hits; // host data
	unsigned int *d_hits; // device data
	unsigned int sum;
	unsigned int nBytes = N*sizeof(unsigned int);
	unsigned int LOOP = 100;

	h_hits = new unsigned int [N];
	/* Allocate memory on device */
	cudaMalloc((void **) &d_hits, nBytes);

	/* Trafer data to device */
	cudaMemcpy(d_hits, h_hits, nBytes, cudaMemcpyHostToDevice);

		
	unsigned int loc_sum = 0;
	for (int j=0; j<LOOP; ++j) {
		loc_sum += dmc_GPU(d_hits, N, M, k);
		MPI_Allreduce(&loc_sum,&sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	}

	printf("Resulting integration: [%Ld]\n", 2.0 * (long double)loc_sum / ((long double) N));

	delete [] h_hits;
	cudaFree(d_hits);
	MPI_Finalize();
	return 0;
}
