#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>
#include <unistd.h>

#define COUNT_OF(x) ((sizeof(x)/sizeof(0[x])))
#define MALLOC_CHECK(x) do { printf("\nMemmory allocation error at {%s}, line [%d]\n", x, __LINE__); exit(-1); } while(0)
#define DIFF_CHECK(x) do { printf("\nTime difference is less than 0 at {%s}, line [%d]\n", x, __LINE__); exit(-1); } while(0)
#define USAGE() do { printf("Usage: omp_ep5 < m > < n > < p > { threads },\n\twith m, p, n, threads all integers!\n"); exit(0); } while(0)
#define DBG(x) printf("\nDBG - %s at LINE [%d]\t", x, __LINE__);

int m, n, p, nthreads = 2;
double ** A, ** B, ** C;

/**
 * Thread routine.
 * Each thread works on a portion of the 'matrix'.
 * The start and end of the portion depend on the 'arg' which
 * is the ID assigned to threads sequentially. 
 */

void * worker(void *arg){
	int i, j, k, tid = *(int *)(arg); // get the thread ID assigned sequentially.;

#ifdef DEBUG
	char dbgmsg[32];
	sprintf(dbgmsg, "Spawned thread %d", tid);
	DBG(dbgmsg);
#endif
	//for (i = tid; i < m; i += nthreads){
	for (i = 0; i < m; i ++){
		for (j = 0; j < n; j++){
			//for (k = tid; k < p; k++){
			for (k = tid; k < p; k += nthreads){
#ifdef DEBUG
	printf("tid{%d} - i:[%d] j:[%d] k:[%d]\n",tid, i, j, k);
#endif
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	return NULL;
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1){
	long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
	result->tv_sec = diff / 1000000;
	result->tv_usec = diff % 1000000;

	return (diff<0);
}

int main(int argc, char *argv[]){
#ifdef DEBUG
	DBG("Enter main");	
#endif
	if(argc == 4) {
		nthreads = sysconf(_SC_NPROCESSORS_ONLN) - 1;
	} else if (argc == 5) {
		nthreads = (int) strtol(argv[4], (char **)NULL, 10); if(!nthreads) USAGE();
	} else {
		USAGE();
	}

	m = (int) strtol(argv[1], (char **)NULL, 10); if(!m) USAGE();
	n = (int) strtol(argv[2], (char **)NULL, 10); if(!n) USAGE(); //printf("n:[%d]\n",n);
	p = (int) strtol(argv[3], (char **)NULL, 10); if(!p) USAGE(); //printf("p:[%d]\n",p);
	A = (double **)malloc(m * sizeof(double *)); if(!A) MALLOC_CHECK("A");
	B = (double **)malloc(p * sizeof(double *)); if(!B) MALLOC_CHECK("B");
	C = (double **)malloc(m * sizeof(double *)); if(!C) MALLOC_CHECK("C");
	for (int i = 0; i < m; i++){
		A[i] = (double *)malloc(p * sizeof(double)); if(!A[i]) MALLOC_CHECK("A[i]");
		C[i] = (double *)malloc(n * sizeof(double)); if(!C[i]) MALLOC_CHECK("C[i]");
	}
	for (int j = 0; j < p; j++){
		B[j] = (double *)malloc(n * sizeof(double)); if(!B[j]) MALLOC_CHECK("B[j]");
	}

	pthread_t * threads = (pthread_t *) malloc(nthreads * sizeof(pthread_t));

#ifdef DEBUG
	DBG("Finish Mem Alloc");
#endif
	struct timeval begin, end;
	gettimeofday(&begin, NULL);
	for (int i = 0; i < nthreads; i++) {
		int *tid;
		tid = (int *) malloc(sizeof(int));
		*tid = i;
	 	pthread_create(&threads[i], NULL, worker, (void *)tid);
	}
#ifdef DEBUG
	DBG("Start thread spawning");
#endif
	//printf("nth{%d} - m [%d], n [%d], p [%d]",nthreads,m,n,p); exit(0);
	for (int j = 0; j < nthreads; j++) {
		pthread_join(threads[j], NULL);
	}
	gettimeofday(&end, NULL);
	
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
