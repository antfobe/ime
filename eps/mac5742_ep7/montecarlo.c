#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PI (3.141592653589793238462643383279502884197169399375105820974)

#define USAGE() do { printf("Usage: ./main < N > < k > < M >\n"); exit(0); } while(0)
int main(int argc, char * argv[]) {
	if(argc != 4) {
		USAGE();
	}

	long N = (long) strtol(argv[1], (char **)NULL, 10); if(!N){ USAGE();}
	int k = (int) strtol(argv[2], (char **)NULL, 10); if(!N){ USAGE();}
	int M = (int) strtol(argv[3], (char **)NULL, 10); if(!N){ USAGE();}
	long hits = 0;
	long double randX, result, f;

	srand(time(NULL));

	for (long i = 0; i < N; i++) {
		randX = ((long double)rand()/(long double)(RAND_MAX)) * .5;
		while (randX == 0.0) randX = ((long double)rand()/(long double)(RAND_MAX)) * .5;
		f = (sin((2 * M + 1) * PI * randX) * 
		     cos(2 * PI * k * randX)) / 
		     sin(PI * randX);
		if (f > 0) hits++;
	}

	result = 2 * ((long double) hits/N);
	printf("Resulting integration is [%Lf], (%ld / %ld)\n", result, hits, N);
	return 0;
}
