#include "ep5.h"

/* OpenMP parallel matrix multiplication function,
 * pretty straight forward - just declared shared
 * and private variables plus schedule, all in 
 * 'vanilla' pragmas */

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
#ifdef DEBUG
	printf("tid{%d} - i:[%ld] j:[%ld] k:[%ld]\n", omp_get_thread_num(), i, j, k);
#endif
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	}   
	/* End of parallel region */
	return 0;
}

/* Function made to get time differential between
 * two system times, as clock difference will 
 * only give processing time, which should be 
 * always greater for parallel operations...*/

int tvsub(struct timeval *result, struct timeval *t2, struct timeval *t1){
	long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
	result->tv_sec = diff / 1000000;
	result->tv_usec = diff % 1000000;

	return (diff<0);
}

/* This one below reads the matrix stored in 
 * 'filename' according to the exercise 
 * especifications. Outputs an allocated and
 * initialized matrix */

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

/* Function used to allocate and initialize
 * with zeros a matrix with dimension 
 * (rows x cols). */

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

/* Below only allocates a matrix with size
 * (rows x cols), different than alloc_initM,
 * which also initializes with zeros.
 * This keeps dirty memmory data in the 
 * allocated matrix */

double ** alloc_onlyM(long rows, long cols){
	double ** M = (double **)malloc(rows * sizeof(double *)); if(!M) MALLOC_CHECK("M, alloc_onlyM");
	for (long i = 0; i < rows; i++){
		M[i] = (double *)malloc(cols * sizeof(double)); if(!M[i]) MALLOC_CHECK("M[i], alloc_onlyM");
	}
	return M;
}

/* freeM was made to free a double (double)
 * array with first dimension size of 
 * 'rows'. No safety checks here: if size
 * is less than 'rows',  a segmentation
 * fault will occur! */

void freeM(long rows, double ** M){
	for (long i = 0; i < rows; i++){
		free(M[i]);
	}
	free(M);
}

/* Writes double (double) array M to a file 
 * named 'filename' (will overwrite!), in
 * accordance to the exercise especifications. */

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
 * The function below defines worker threads 
 * behavior:
 * Each thread works on a portion of the matrixes
 * (A and B). The start and end of the portion 
 * depend on the 'arg' which is the ID assigned 
 * to threads sequentially, for instance:
 * given 4 threads, thread 0 will work A[*][0] * 
 * B[0][*], A[*][4] * B[4][*], ..., A[*][k] * 
 * B[k][*], k < p; */

void * pth_mworker(void *arg){
	long i, j, k;
	int tid = *(int *)(arg); /* get the thread ID, was assigned sequentially */

#ifdef DEBUG
	char dbgmsg[32];
	sprintf(dbgmsg, "Spawned thread %d", tid);
	DBG(dbgmsg);
#endif
	for (i = 0; i < m; i ++){
		for (j = 0; j < n; j++){
			for (k = (long) tid; k < p; k += nthreads){
#ifdef DEBUG
	printf("tid{%d} - i:[%ld] j:[%ld] k:[%ld]\n",tid, i, j, k);
#endif
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	return NULL;
}
