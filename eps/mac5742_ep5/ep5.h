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

int nthreads = 2;
long m = 0, n = 0, p = 0;
double ** A, ** B, ** C; 
struct timeval begin, end;

int omp_mmul(long m, long n, long p, double ** A, double ** B, double ** C);
int tvsub(struct timeval *result, struct timeval *t2, struct timeval *t1);
double ** readM(char * filename);
char * writeC(char * filename, double ** C);
void * pth_mworker(void *arg);
int mmul(int m, int n, int p, double ** a, double ** b, double ** c);
