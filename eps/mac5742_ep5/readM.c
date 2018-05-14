#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MALLOC_CHECK(x) do { printf("\nMemmory allocation error at {%s}, line [%d]\n", x, __LINE__); exit(-1); } while(0)
#define DIFF_CHECK(x) do { printf("\nTime difference is less than 0 at {%s}, line [%d]\n", x, __LINE__); exit(-1); } while(0)
#define USAGE() do { printf("Usage: omp_ep5 < m > < n > < p >,\n\twith m, p, n integers!\n"); exit(0); } while(0)
#define	DBG(x) printf("\nDBG - %s at LINE [%d]\n", x, __LINE__);

long m = 0, n = 0, p = 0;

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

int main(int argc, char ** argv){
#ifdef DEBUG
	DBG("Enter main");	
#endif
	if(argc != 2) {
		USAGE();
	}
	double ** A = readM(argv[1]);
	A = readM(argv[1]);
	if(!writeC("C.txt", A)) {printf("failed to write C to file at %d", __LINE__); exit(-1);}
#ifdef DEBUG
	DBG("Done reading M from file");	
#endif
	for(long i = 0; i < m; i ++) {
		for(long j = 0; j < p; j ++)
#ifdef DEBUG
	printf("A[%ld][%ld] = {%lf}\n", i, j, A[i][j]);
#endif
	free(A[i]);
	}
	free(A);
	return 0;
}
