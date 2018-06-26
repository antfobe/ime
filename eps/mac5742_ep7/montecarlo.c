#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PI (3.141592653589793238462643383279502884197169399375105820974)

int main() {
	int i, throws = 99999, circleDarts = 0, M = 512, k = 32;
	long double randX, randY, result, f;

	srand(time(NULL));

	for (i = 0; i < throws; i++) {
		randX = ((long double)rand()/(long double)(RAND_MAX)) * .5;
		while (randX == 0.0) randX = ((long double)rand()/(long double)(RAND_MAX)) * .5;
		f = (sin((2 * M + 1) * PI * randX) * 
		     cos(2 * PI * k * randX)) / 
		     sin(PI * randX);
		if (f > 0) circleDarts++;
	}

	result = 2 * ((long double) circleDarts/throws);
	printf("Resulting integration is [%Lf], (%ld / %ld)\n", result, circleDarts, throws);
	return 0;
}
