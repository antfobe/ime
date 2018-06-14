#include <stdio.h>
#include <math.h>

#define N 3
#define N2 N * N

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

long mnum = 0;

__global__
void minappl(int * arrayM, int * arrayS, int mnum){
	extern __shared__ int smin[N2];
	unsigned int tid = threadIdx.x;
	unsigned int i = (blockIdx.x*blockDim.x + threadIdx.x);
//	for(int a = 0; a < N * N * mnum; a ++)printf("arrayM[%d]: (%d)\n", a, arrayM[a]);
	if (i < mnum) {
		/* each thread loads one element from global to shared mem */
		for (unsigned int j = 0; j < N * N; j++) {
			smin[j] = arrayM[(i % mnum) * N * N + j];
		}
		__syncthreads();
		
		/* do reduction in shared mem */
		for (unsigned int s = 1; s < blockDim.x; s++) {
			if (tid < s) {
				for (unsigned int j = 0; j < N * N; j++){
					smin[j] = min(smin[j], arrayM[((i + s) % mnum) * N * N + j]);
				}
			}
			__syncthreads();	
		}
		/* write block result to global mem */
	if (i == 0)
		for (unsigned int j = 0; j < N * N; j++)
				arrayS[j] = smin[j];
	}
}

int * file2buffer(char * filename) {
	
	FILE * fp = fopen(filename, "r"); if(!fp) {printf("fopen failed at %d", __LINE__); exit(-1);}

	fscanf(fp, "%ld", &mnum);
	int * M = (int *)malloc(mnum * N * N * sizeof(int));

	for(long i = 0; i < mnum; i++){
		fscanf(fp, "%s", stdout);
//		printf("stdout: [%s]\n", stdout);
		for(int l, m, n, o = 0; fscanf(fp, "%d %d %d", &l, &m, &n) == 3; o++){
			M[N * (o + N * i)] = l; 
			M[N * (o + N * i) + 1] = m;
			M[N * (o + N * i) + 2] = n;
//			printf("l, m, n: (%d, %d, %d)\n", l, m, n);
		}
	}
	fclose(fp);
	return M;
}

int main(int argc, char * argv[]){

	if(argc != 2) {
		printf("Usage: ./ep6 < filename >\n"); exit(0);
	}

	int * M = file2buffer(argv[1]);
	int S[N2];
	int * dM; gpuErrchk(cudaMalloc(&dM, mnum*N*N*sizeof(int)));
	int * dS; gpuErrchk(cudaMalloc(&dS, N*N*sizeof(int)));

	/* Initialize S */
	for (int i = 0; i < N * N; i++){
		S[i] = M[i];
//		printf("S[%d][%d]: %d\n", i / N, i % N, S[i]);
	}

	gpuErrchk(cudaMemcpy(dM, M, mnum*N*N*sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dS, S, N*N*sizeof(int), cudaMemcpyHostToDevice));
	minappl<<<N * N, mnum * N * N>>>(dM, dS, mnum);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk(cudaMemcpy(S, dS, N*N*sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N * N; i++)
		printf("S[%d][%d]: %d\n", i / N + 1, i % N + 1, S[i]);

	gpuErrchk(cudaFree(dS));
	gpuErrchk(cudaFree(dM));
	free(M);
}
