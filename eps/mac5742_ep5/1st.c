#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>
#include <pthread.h>

#define MALLOC_CHECK(x) do { printf("\nMemmory allocation error at {%s}, line [%d]\n", x, __LINE__); exit(-1); } while(0)
#define DIFF_CHECK(x) do { printf("\nTime difference is less than 0 at {%s}, line [%d]\n", x, __LINE__); exit(-1); } while(0)
#define USAGE() do { printf("Usage: omp_ep5 < m > < n > < p >,\n\twith m, p, n integers!\n"); exit(0); } while(0)
#define DBG(x) printf("\nDBG - %s at LINE [%d]\t", x, __LINE__);

long m = 0, n = 0, p = 0, nthreads = 2;
double ** A, ** B, ** C; 

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
	printf("tid{%d} - i:[%d] j:[%d] k:[%d]\n", omp_get_thread_num(), i, j, k);
#endif
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
	double ** M = (double **)malloc(rows * sizeof(double *)); if(!M) MALLOC_CHECK(filename);
	for (long i = 0; i < rows; i++){
		M[i] = (double *)malloc(cols * sizeof(double)); if(!M[i]) MALLOC_CHECK(filename);
		for (long j = 0; j < cols; j++){
			M[i][j] = 0;
		}
	}
	double value;
	for(long i, j; fscanf(fp, "%ld %ld %lf", &i, &j, &value) == 3; ){
		M[i - 1][j - 1] = value;
	}
	fclose(fp);
	return M;
}

char * writeC(char * filename, double ** C){
#ifdef DEBUG
	DBG("Enter writeC"); 
#endif
	FILE * fp = fopen(filename, "w"); if(!fp) {printf("fopen failed at %d", __LINE__); exit(-1);}
	fprintf(fp, "%ld %ld\n", m, n);
	for(long i = 0; i < m; i++)
		for(long j = 0; j < n; j++)
			fprintf(fp, "%ld %ld %lf\n", i + 1, j + 1, C[i][j]);
	fclose(fp);
	return filename;	
}

/**
 * Thread routine.
 * Each thread works on a portion of the 'matrix'.
 * The start and end of the portion depend on the 'arg' which
 * is the ID assigned to threads sequentially. 
 */

void * worker(void *arg){
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
#ifdef DEBUG
	printf("tid{%d} - i:[%d] j:[%d] k:[%d]\n",tid, i, j, k);
#endif
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	return NULL;
}

int main(int argc, char ** argv){
#ifdef DEBUG
	DBG("Enter main");	
#endif
	struct timeval begin, end;
	char * outfile = "C.txt";
	pthread_t * threads;
	switch(argv[1][0]){
		case 'p':
				nthreads = sysconf(_SC_NPROCESSORS_ONLN);
			if(!argv[2] || !argv[3]){
				USAGE();
			} else {
				A = readM(argv[2]);
#ifdef DEBUG
	DBG("Done reading A from file");	
#endif
				B = readM(argv[3]);
#ifdef DEBUG
	DBG("Done reading B from file");	
#endif
				threads = (pthread_t *) malloc(nthreads * sizeof(pthread_t));
#ifdef DEBUG
	DBG("Finish Mem Alloc");
#endif
			}
			if(argv[3]){
				outfile = argv[3];
			}
#ifdef DEBUG
	DBG("Start thread spawning");
#endif
			gettimeofday(&begin, NULL);
			for (int i = 0; i < nthreads; i++) {
				int *tid;
				tid = (int *) malloc(sizeof(int));
				*tid = i;
				pthread_create(&threads[i], NULL, worker, (void *)tid);
			}
			for (long j = 0; j < nthreads; j++) {
				pthread_join(threads[j], NULL);
			}
			gettimeofday(&end, NULL);
			
			for (long i = 0; i < m; i++){
				free(A[i]);
				free(C[i]);
			}
			free(A);
			free(C);

			for (long j = 0; j < p; j++){
				free(B[j]);
			}
			free(B);
			free(threads);

			if(!writeC(outfile, A)) {DBG("failed to write C matrix to file"); exit(-1);}
		break;
		case 'o':
			if(!argv[2] || !argv[3]){
				USAGE();
			} else {
				A = readM(argv[2]);
#ifdef DEBUG
	DBG("Done reading A from file");	
#endif
				B = readM(argv[3]);
#ifdef DEBUG
	DBG("Done reading B from file");	
#endif
			}
			if(argv[3]){
				outfile = argv[3];
			}
			
			gettimeofday(&begin, NULL);
			omp_mmul(m, n, p, A, B, C);
			gettimeofday(&end, NULL);
#ifdef DEBUG
	for (long i = 0; i < m; i++)
		for (long j = 0; j < p; j++)
			printf("C[%ld][%ld] = {%lf}\n", i, j, C[i][j]);
#endif
			if(!writeC(outfile, C)) {DBG("failed to write C matrix to file"); exit(-1);}

			for (long i = 0; i < m; i++){
				free(A[i]);
				free(C[i]);
			}
			free(A);
			free(C);
			for (long j = 0; j < p; j++){
				free(B[j]);
			}
			free(B);
		break;
		case 'd':
			m = (long) strtol(argv[2], (char **)NULL, 10); if(!m) USAGE();
			n = (long) strtol(argv[3], (char **)NULL, 10); if(!n) USAGE();
			p = (long) strtol(argv[4], (char **)NULL, 10); if(!p) USAGE();
			A = (double **)malloc(m * sizeof(double *)); if(!A) MALLOC_CHECK("A");
			B = (double **)malloc(p * sizeof(double *)); if(!B) MALLOC_CHECK("B");
			C = (double **)malloc(m * sizeof(double *)); if(!C) MALLOC_CHECK("C");
			for (long i = 0; i < m; i++){
				A[i] = (double *)malloc(p * sizeof(double)); if(!A[i]) MALLOC_CHECK("A[i]");
				C[i] = (double *)malloc(n * sizeof(double)); if(!C[i]) MALLOC_CHECK("C[i]");
			}
			for (long j = 0; j < p; j++){
				B[j] = (double *)malloc(n * sizeof(double)); if(!B[j]) MALLOC_CHECK("B[j]");
			}
#ifdef DEBUG
	DBG("Finish Mem Alloc");
#endif
			gettimeofday(&begin, NULL);
			omp_mmul(m, n, p, A, B, C);
			gettimeofday(&end, NULL);
			for (long i = 0; i < m; i++){
				free(A[i]);
				free(C[i]);
			}
			free(A);
			free(C);
			for (long j = 0; j < p; j++){
				free(B[j]);
			}
			free(B);
		break;
		case 'D':
			if(argc == 5) {
				nthreads = sysconf(_SC_NPROCESSORS_ONLN) - 1;
			} else if (argc == 6) {
				nthreads = (int) strtol(argv[5], (char **)NULL, 10); if(!nthreads) USAGE();
			} else {
				USAGE();
			}
			pthread_t * threads = (pthread_t *) malloc(nthreads * sizeof(pthread_t));
			m = (long) strtol(argv[2], (char **)NULL, 10); if(!m) USAGE();
			n = (long) strtol(argv[3], (char **)NULL, 10); if(!n) USAGE();
			p = (long) strtol(argv[4], (char **)NULL, 10); if(!p) USAGE();
			A = (double **)malloc(m * sizeof(double *)); if(!A) MALLOC_CHECK("A");
			B = (double **)malloc(p * sizeof(double *)); if(!B) MALLOC_CHECK("B");
			C = (double **)malloc(m * sizeof(double *)); if(!C) MALLOC_CHECK("C");
			for (long i = 0; i < m; i++){
				A[i] = (double *)malloc(p * sizeof(double)); if(!A[i]) MALLOC_CHECK("A[i]");
				C[i] = (double *)malloc(n * sizeof(double)); if(!C[i]) MALLOC_CHECK("C[i]");
			}
			for (long j = 0; j < p; j++){
				B[j] = (double *)malloc(n * sizeof(double)); if(!B[j]) MALLOC_CHECK("B[j]");
			}
#ifdef DEBUG
	DBG("Finish Mem Alloc");
#endif
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
			for (int j = 0; j < nthreads; j++) {
				pthread_join(threads[j], NULL);
			}
			gettimeofday(&end, NULL);
			
			for (long i = 0; i < m; i++){
				free(A[i]);
				free(C[i]);
			}
			free(A);
			free(C);

			for (long j = 0; j < p; j++){
				free(B[j]);
			}
			free(B);
			free(threads);
		break;
		default:
			USAGE();
		break;
	}
	
	if(tvsub(&end, &end, &begin)) DIFF_CHECK("timeval");
	printf("(%s): Matrix multiplication took [%ld.%06ld] seconds\n", __FILE__, end.tv_sec, end.tv_usec);

	return 0;
}
