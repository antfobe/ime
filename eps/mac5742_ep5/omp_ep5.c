#include <omp.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

//#define COUNT_OF(x) ((sizeof(x)/sizeof(0[x])) / ((size_t)(!(sizeof(x) % sizeof(0[x])))))
#define COUNT_OF(x) ((sizeof(x)/sizeof(0[x])))
#define MALLOC_CHECK(x) do { printf("\nMemmory allocation error at {%s}, line [%d]\n", x, __LINE__); exit(-1); } while(0)
#define DIFF_CHECK(x) do { printf("\nTime difference is less than 0 at {%s}, line [%d]\n", x, __LINE__); exit(-1); } while(0)
#define USAGE() do { printf("Usage: omp_ep5 < m > < n > < p >,\n\twith m, p, n integers!\n"); exit(0); } while(0)

int omp_mmul(int m, int n, int p, double ** a, double ** b, double ** c){
	int i, j, k;
	#pragma omp parallel shared(a, b, c) private(i, j, k)
	{

	/* Do matrix multiply sharing iterations on outer loop */

	#pragma omp for schedule (static)
	for (i = 0; i < m; i++){
		for(j = 0; j < n; j++){
			for (k = 0; k < p; k++){
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	}   
	/* End of parallel region */
	return 0;
}

int mmul(int m, int n, int p, double ** a, double ** b, double ** c){
	int i,j,k;
	for (i = 0; i < m; i++){
		for (j = 0; j < n; j++){
			for (k = 0; k < p; k++){
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	return 0;
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1){
	long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
	result->tv_sec = diff / 1000000;
	result->tv_usec = diff % 1000000;

	return (diff<0);
}

int main(int argc, char ** argv){
	if(argc != 4) {
		USAGE();
	}

	int m = (int) strtol(argv[1], (char **)NULL, 10); if(!m) USAGE();
	int n = (int) strtol(argv[2], (char **)NULL, 10); if(!n) USAGE(); //printf("n:[%d]\n",n);
	int p = (int) strtol(argv[3], (char **)NULL, 10); if(!p) USAGE(); //printf("p:[%d]\n",p);
	double ** A = (double **)malloc(m * sizeof(double *)); if(!A) MALLOC_CHECK("A");
	double ** B = (double **)malloc(p * sizeof(double *)); if(!B) MALLOC_CHECK("B");
	double ** C = (double **)malloc(m * sizeof(double *)); if(!C) MALLOC_CHECK("C");
	for (int i = 0; i < m; i++){
		A[i] = (double *)malloc(p * sizeof(double)); if(!A[i]) MALLOC_CHECK("A[i]");
		C[i] = (double *)malloc(n * sizeof(double)); if(!C[i]) MALLOC_CHECK("C[i]");
	}
	for (int j = 0; j < p; j++){
		B[j] = (double *)malloc(n * sizeof(double)); if(!B[j]) MALLOC_CHECK("B[j]");
	}
	
	struct timeval begin, end;
	gettimeofday(&begin, NULL); //printf("%s\n", asctime(localtime(time(NULL))));
	omp_mmul(m, n, p, A, B, C);
	//mmul(m, n, p, A, B, C);
	gettimeofday(&end, NULL); //printf("%s\n", asctime(localtime((time_t *)time(NULL)))); 

	for (int i = 0; i < m; i++){
		free(A[i]);
		free(C[i]);
	}
	free(A);
	free(C);

	for (int j = 0; j < p; j++){
		free(B[j]);
	}
	free(B);

	if(timeval_subtract(&end, &end, &begin)) DIFF_CHECK("timeval");
	printf("(%s): Matrix multiplication took [%ld.%06ld] seconds\n", __FILE__, end.tv_sec, end.tv_usec);

	return 0;
}
