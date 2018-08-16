#include <stdio.h>
#include <math.h>

#define N 3
#define N2 N * N
#define BLOCK_SIZE 512

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess){
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

long mnum = 0;

__global__
void minappl(int * arrayM, int mnum){
	extern __shared__ int smin[N2];
	unsigned int i = (blockIdx.x*blockDim.x + threadIdx.x);
	if (i < mnum) {
		/* do reduction in shared mem */
		for (unsigned int offset = 1; offset <= mnum; offset <<= 1) {
			if (i + offset < mnum)
				for (unsigned int j = 0; j < N * N; j++){
					arrayM[i * N * N + j] = min(arrayM[i * N * N + j], arrayM[(i + offset) * N * N + j]);
				}
//			printf("> i, offset, min: [%d,%d] {%d}\n", i, offset, arrayM[i * N * N]);
		}
		__syncthreads();
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
	int * dM; gpuErrchk(cudaMalloc(&dM, mnum*N*N*sizeof(int)));

	gpuErrchk(cudaMemcpy(dM, M, mnum*N*N*sizeof(int), cudaMemcpyHostToDevice));
	minappl<<<(N * N * mnum) / BLOCK_SIZE, BLOCK_SIZE>>>(dM, mnum);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk(cudaMemcpy(M, dM, N*N*sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < N * N; i++)
		printf("S[%d][%d]: %d\n", i / N + 1, i % N + 1, M[i]);

	gpuErrchk(cudaFree(dM));
	free(M);
	
	return 0;
}
