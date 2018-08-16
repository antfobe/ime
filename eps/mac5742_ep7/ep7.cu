#include <omp.h>
#include <mpi.h>
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
__global__ void mc_integration(int seed, int threadwork, double * d_hits, unsigned int N, int M, int k, double * f2){
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
			f2[idx] += (f * f); 
			d_hits[idx] += f;
			__syncthreads();
			}
	__syncthreads();
}

__global__ void sum_reduction(double * array, long N){
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	for(long i = 1; i < N; i <<= 1) {
		array[idx] += array[idx + i];
		__syncthreads();
	}
	__syncthreads();
}

double mc_integrationGPU(unsigned int N, int M, int k, double * f2){
	double * d_hits; gpuErrchk(cudaMalloc((void **)&d_hits, N * sizeof(double)));
	double * d_f2; gpuErrchk(cudaMalloc((void **)&d_f2, N * sizeof(double)));
	double h_hits = 0;
	curandState *d_state; gpuErrchk(cudaMalloc(&d_state, sizeof(curandState)));

	//setup_kernel<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(time(NULL), d_state);
	int threadwork = 4096;
	mc_integration<<<(N / (BLOCK_SIZE * threadwork)) | 1 , BLOCK_SIZE>>>(time(NULL), threadwork, d_hits, N, M, k, d_f2);
	gpuErrchk(cudaPeekAtLastError());

	sum_reduction<<<(N / BLOCK_SIZE) | 1, BLOCK_SIZE>>>(d_hits, N);
	gpuErrchk(cudaPeekAtLastError());
	
	gpuErrchk(cudaMemcpy(&h_hits, d_hits, sizeof(double), cudaMemcpyDeviceToHost));

	sum_reduction<<<(N / BLOCK_SIZE) | 1, BLOCK_SIZE>>>(d_f2, N);
	gpuErrchk(cudaPeekAtLastError());
	
	gpuErrchk(cudaMemcpy(&f2[0], d_hits, sizeof(double), cudaMemcpyDeviceToHost));
	
	gpuErrchk(cudaFree(d_hits));
	gpuErrchk(cudaFree(d_f2));
	gpuErrchk(cudaFree(d_state));

	return h_hits;
}

double montecarlo_OMP(unsigned int N, unsigned int k, unsigned int M, double * f2){
	double randX, f = 0.0, f_hits = 0.0;
	unsigned int i =0;
	//unsigned int r_hits = 0, i =0;
	srand(time(NULL));
	omp_set_dynamic(0);
	omp_set_num_threads(omp_get_max_threads());
	#pragma omp parallel shared(f_hits, f2[0]) private(i, randX, f)
	{
	printf("OpenMP running with (%d/%d) threads\n", omp_get_num_threads(), omp_get_max_threads());
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
 
double montecarlo_singleCPU(unsigned int N, unsigned int k, unsigned int M, double * f2){
	double randX, f;
	double r_hits = 0;
	srand(time(NULL));

	for (unsigned int i = 0; i < N; i++) {
		randX = ((double)rand()/(double)(RAND_MAX)) * .5;
		while (randX == 0.0) randX = ((double)rand()/(double)(RAND_MAX)) * .5;
		f = (sin((2 * M + 1) * PI * randX) * 
		     cos(2 * PI * k * randX)) / 
		     sin(PI * randX);
		f2[0] += (f * f);
		r_hits += f;
	}
	return r_hits;
}

double tvdiff(struct timeval * t2, struct timeval * t1){
	long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
	long double result = diff / 1000000;
	result += (long double) (diff % 1000000) / 1000000;

	if (diff < 0) return (-1);
	else return (double) result;
}

int main(int argc, char * argv[]) {
	if(argc != 4) {
		USAGE();
	}

	unsigned int N = (unsigned int) strtol(argv[1], (char **)NULL, 10); if(!N){ USAGE();}
	int k = (int) strtol(argv[2], (char **)NULL, 10); if(!k){ USAGE();}
	int M = (int) strtol(argv[3], (char **)NULL, 10); if(!M){ USAGE();}
	long double result = 0.0, hits, err = 0.0;
	double f2[1], cpu_time = 0.0, gain; f2[0] = 0.0; 
	struct timeval begin, end, gpu_b, gpu_e;
	
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

	if (!world_rank){	
		/* Single thread */
		gettimeofday(&begin, NULL);
		hits = (long double)montecarlo_singleCPU(N, k, M, f2);
		gettimeofday(&end, NULL);
		err = sqrtl(abs((f2[0]/N - hits * hits/N) / N));
		result = (long double) hits / N;
		cpu_time = tvdiff(&end, &begin);
		printf("Sequencial Time (1 thread): (%lf)s\nResulting integration is [%Lf]\nError Interval: [%Lf,%Lf]\n\n", cpu_time, result, result - (long double) err, result + (long double) err);
		err = result = 0.0;

		/* OpenMP multithread */
		gettimeofday(&begin, NULL);
		hits = (long double)montecarlo_OMP(N, k, M, f2);
		gettimeofday(&end, NULL);
		err = sqrtl(abs((f2[0]/N - hits * hits/N) / N));
		result = (long double) hits / N;
		printf("CPU Time with muliple threads: (%lf)s\nResulting integration is [%Lf]\nError Interval: [%Lf,%Lf]\n\n", tvdiff(&end, &begin), result, result - (long double) err, result + (long double) err);
		err = result = 0.0;
	}

	/* GPU + 1 CPU thread */
	if (world_rank == 1) {
    		MPI_Recv(&hits, 1, MPI_LONG_DOUBLE, world_rank - 1, 0,
            	MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
		gettimeofday(&begin, NULL);
		hits += (long double)montecarlo_singleCPU(N / 2, k, M, f2);
	} else  if (world_rank == 0){
    	// Set the hits's (hits) value if you are process 0
		hits = 0.0;	
		gettimeofday(&gpu_b, NULL);
		hits += (long double)mc_integrationGPU(N / 2, M, k, f2);
		gettimeofday(&gpu_e, NULL);
		
	}

	MPI_Send(&hits, 1, MPI_LONG_DOUBLE, (world_rank + 1) % world_size,
         0, MPI_COMM_WORLD);

	// Now process 0 can receive from the last process.
	if (world_rank == 0) {
		MPI_Recv(&hits, 1, MPI_LONG_DOUBLE, world_size - 1, 0,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		gettimeofday(&end, NULL);
		err = sqrtl(abs((f2[0]/N - hits * hits/N) / N));
		result = (long double) hits / N;
		printf("GPU and single CPU Time (N/2 for each): (%lf)s\nResulting integration is [%Lf]\nError Interval: [%Lf,%Lf]\n\n", tvdiff(&end, &begin), result, result - (long double) err, result + err);
		err = result = 0.0;

		gain = cpu_time/tvdiff(&gpu_e, &gpu_b);
	}

	/* Balance GPU + n CPU */
	if (world_rank != 0) {
    		MPI_Recv(&hits, 1, MPI_LONG_DOUBLE, world_rank - 1, 0,
            	MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
		gettimeofday(&begin, NULL);
		if (gain < 1) {
			/* Careful not to divide by 0, world_size >= 2 */
			//hits += (long double)montecarlo_OMP(N / (world_size - 1), k, M, f2);
			hits += (long double)montecarlo_singleCPU(N / (world_size - 1), k, M, f2);
		} else {
			hits += (long double)montecarlo_OMP(N / ((gain + 1) * (world_size - 1)), k, M, f2);
		}	
	} else  {
    	// Set the hits's (hits) value if you are process 0
		hits = 0.0;	
		if(gain >= 1) {
			if(gpu_bite > 0){
				for (unsigned int n = N % gpu_bite; n < (N * gain) / ((gain + 1) * (world_size - 1)); n += gpu_bite){
					hits += mc_integrationGPU(gpu_bite, M, k, f2);
				}
			} else {
				hits += mc_integrationGPU((N * gain) / ((gain + 1) * (world_size - 1)), M, k, f2);
			}
		}
	}

	MPI_Send(&hits, 1, MPI_LONG_DOUBLE, (world_rank + 1) % world_size,
         0, MPI_COMM_WORLD);

	// Now process 0 can receive from the last process.
	if (world_rank == 0) {
		MPI_Recv(&hits, 1, MPI_LONG_DOUBLE, world_size - 1, 0,
		MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		gettimeofday(&end, NULL);
		err = sqrtl(abs((f2[0]/N - hits * hits/N) / N));
		result = (long double) hits / N;
		printf("Balanced Time: (%lf)s\nResulting integration is [%Lf]\nError Interval: [%Lf,%Lf]\n\n", tvdiff(&end, &begin), result, result - (long double) err, result + err);
		err = result = 0.0;
	}
	MPI_Finalize();
	return 0;
}
