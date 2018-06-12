#include <stdio.h>
#include <stdlib.h>

#define N 3

int * file2buffer(char * filename) {
	
	long mnum;
	FILE * fp = fopen(filename, "r"); if(!fp) {printf("fopen failed at %d", __LINE__); exit(-1);}

	fscanf(fp, "%ld", &mnum);
	int * M = (int *)malloc(mnum * N * N * sizeof(int));

	for(long i = 0; i < mnum; i++){
		fscanf(fp, "%s", stdout);
		for(int l, m, n, o = 0; fscanf(fp, "%d %d %d", &l, &m, &n) == 3; o++){
			M[N * (o + N * i)] = l; 
			M[N * (o + N * i) + 1] = m;
			M[N * (o + N * i) + 2] = n;
		}
	}
	fclose(fp);
	return M;
}

int main(){
	int * M = file2buffer("sample.txt");
	for(int i = 1; i < 2 * N * N + 1; i++)
		if (!(i % 3)){
			printf("[%d]\n", M[i - 1]);
		} else {
			printf("[%d]\t", M[i - 1]);
		}
	free(M);

	return 0;
}
