#include <stdio.h>
#include <math.h>

#define N 3
#define N2 N * N

long mnum = 0;

__global__
void minappl(int * arrayM, int * arrayS){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	arrayS[i % 9] = min(arrayS[i % 9], arrayM[i]);
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
	int * dM; cudaMalloc(&dM, mnum*N*N*sizeof(int));
	int * dS; cudaMalloc(&dS, N*N*sizeof(int));

	/* Initialize S */
	for (int i = 0; i < N * N; i++){
		S[i] = M[i];
//		printf("S[%d][%d]: %d\n", i / N, i % N, S[i]);
	}

	cudaMemcpy(dM, M, mnum*N*N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dS, S, N*N*sizeof(int), cudaMemcpyHostToDevice);
	minappl<<<N * N, mnum * N * N>>>(dM, dS);
	cudaMemcpy(S, dS, N*N*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N * N; i++)
		printf("S[%d][%d]: %d\n", i / N + 1, i % N + 1, S[i]);

	cudaFree(dS);
	cudaFree(dM);
	free(M);
}
