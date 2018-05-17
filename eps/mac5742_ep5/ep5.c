#include "ep5.h"

int omp_mmul(long m, long n, long p, double ** A, double ** B, double ** C){
	long i, j, k;
#ifdef DEBUG
	DBG("Entering parallel region")
#endif
	#pragma omp parallel shared(A, B, C) private(i, j, k)
	{

	/* Do matrix multiply sharing iterations on outer loop */

	#pragma omp for schedule (static)
	for (i = 0; i < m; i++){
		for(j = 0; j < n; j++){
			for (k = 0; k < p; k++){
//#ifdef DEBUG
//	printf("tid{%d} - i:[%d] j:[%d] k:[%d]\n", omp_get_thread_num(), i, j, k);
//#endif
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	}   
	/* End of parallel region */
	return 0;
}

int tvsub(struct timeval *result, struct timeval *t2, struct timeval *t1){
	long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
	result->tv_sec = diff / 1000000;
	result->tv_usec = diff % 1000000;

	return (diff<0);
}

double ** readM(char * filename) {
#ifdef DEBUG
	DBG("Enter readM");	
#endif
	long rows, cols;
	FILE * fp = fopen(filename, "r"); if(!fp) {printf("fopen failed at %d", __LINE__); exit(-1);}

	fscanf(fp, "%ld %ld", &rows, &cols);
	if(p == rows) {
		n = cols;
	} else {
		m = rows; p = cols;
	}
	double ** M = alloc_initM(rows, cols);
	double value;
	for(long i, j; fscanf(fp, "%ld %ld %lf", &i, &j, &value) == 3; ){
		M[i - 1][j - 1] = value;
	}
	fclose(fp);
	return M;
}

double ** alloc_initM(long rows, long cols){
	double ** M = (double **)malloc(rows * sizeof(double *)); if(!M) MALLOC_CHECK("M, alloc_initM");
	for (long i = 0; i < rows; i++){
		M[i] = (double *)malloc(cols * sizeof(double)); if(!M[i]) MALLOC_CHECK("M[i], alloc_initM");
		for (long j = 0; j < cols; j++){
			M[i][j] = 0;
		}
	}
	return M;
}

double ** alloc_onlyM(long rows, long cols){
	double ** M = (double **)malloc(rows * sizeof(double *)); if(!M) MALLOC_CHECK("M, alloc_onlyM");
	for (long i = 0; i < rows; i++){
		M[i] = (double *)malloc(cols * sizeof(double)); if(!M[i]) MALLOC_CHECK("M[i], alloc_onlyM");
	}
	return M;
}

void freeM(long rows, double ** M){
	for (long i = 0; i < rows; i++){
		free(M[i]);
	}
	free(M);
}

char * writeM(char * filename, double ** M){
#ifdef DEBUG
	DBG("Enter writeC"); 
#endif
	FILE * fp = fopen(filename, "w"); if(!fp) {printf("fopen failed at %d", __LINE__); exit(-1);}
	fprintf(fp, "%ld %ld\n", m, n);
	for(long i = 0; i < m; i++)
		for(long j = 0; j < n; j++)
			fprintf(fp, "%ld %ld %lf\n", i + 1, j + 1, M[i][j]);
	fclose(fp);
	return filename;	
}

/**
 * Thread routine.
 * Each thread works on a portion of the 'matrix'.
 * The start and end of the portion depend on the 'arg' which
 * is the ID assigned to threads sequentially. 
 */

void * pth_mworker(void *arg){
	long i, j, k;
	int tid = *(int *)(arg); // get the thread ID assigned sequentially.;

#ifdef DEBUG
	char dbgmsg[32];
	sprintf(dbgmsg, "Spawned thread %d", tid);
	DBG(dbgmsg);
#endif
	/*for (i = tid; i < m; i += nthreads){*/
	for (i = 0; i < m; i ++){
		for (j = 0; j < n; j++){
			/*for (k = tid; k < p; k++){*/
			for (k = (long) tid; k < p; k += nthreads){
//#ifdef DEBUG
//	printf("tid{%d} - i:[%d] j:[%d] k:[%d]\n",tid, i, j, k);
//#endif
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	return NULL;
}
