#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>
#include <pthread.h>

#define MALLOC_CHECK(x) do { printf("\nMemmory allocation error at {%s}, line [%d]\n", x, __LINE__); exit(-1); } while(0)
#define DIFF_CHECK(x) do { printf("\nTime difference is less than 0 at {%s}, line [%d]\n", x, __LINE__); exit(-1); } while(0)
#define VARNAME(x)  #x
#define USAGE() do { printf("Usage: ep5 < implementação: p | o > < path A: 'A.txt' > < path B: 'B.txt'> < path C: 'C.txt' > | \n  < Dry run (on dirty mem): P | O > < m > < n > < p >\n\twith m, n, p integers!\n"); exit(0); } while(0)
#define DBG(x) printf("\nDBG - %s at LINE [%d]: FILE [%s]\n", x, __LINE__, __FILE__);

extern int nthreads;
extern long m, n, p;
extern double ** A, ** B, ** C; 
extern struct timeval begin, end;

int omp_mmul(long m, long n, long p, double ** A, double ** B, double ** C);
int tvsub(struct timeval *result, struct timeval *t2, struct timeval *t1);
double ** readM(char * filename);
double ** alloc_initM(long rows, long cols);
double ** alloc_onlyM(long rows, long cols);
void freeM(long rows, double ** M);
char * writeM(char * filename, double ** C);
void * pth_mworker(void *arg);
int mmul(int m, int n, int p, double ** a, double ** b, double ** c);
