/*
 * COMPILE gcc -lbsd -O0 -std=c99 -o spec_perf speculation_performance.c
 *
 * */

#include <bsd/stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)

#define ITER_MAX 100000

void getHomeopt() {
	const char *home_dir;

	home_dir = getenv("HOME");
	if (likely(home_dir))
		printf("home directory: %s\n", home_dir);
	else
		perror("getenv");
}

void getHome() {
	const char *home_dir;

	home_dir = getenv("HOME");
	if (home_dir)
		home_dir += 3;
	else
		perror("getenv");
}

int likelyLastout(long i, long upto) {
		clock();
	if (likely(i < (upto % ITER_MAX))){
		return likelyLastout(i + 1, upto);
	} else { 
		return i;
	}
}

int unlikelyLastout(long i, long upto) {
	clock();
	if (unlikely(i < (upto % ITER_MAX))){
		return unlikelyLastout(i + 1, upto);
	} else  {
		return i;
	}
}

int normLastout(long i, long upto){
		clock();
	if (i > (upto % ITER_MAX)) {
		return i;
	} else {
		return (normLastout(i + 1, upto));
	}
}

int main(int argc, char ** argv) {
	int iter = 0;
	long randomInt;
	if(argc > 2) {
		/*Remember, i must be < ITER_MAX*/
		iter = (int) strtol(argv[1], (char **)NULL, 10);
		randomInt = (int) strtol(argv[2], (char **)NULL, 10);
	} else {
		char randomData[16];
		arc4random_buf(randomData, sizeof randomData);
		randomInt = (long) randomData;
	}

	clock_t begin = clock();
	getHomeopt();
	clock_t end = clock();
	printf("Getting HOME path from env (non-optimized) took\t[%.06f]s\n", (double)(end - begin) / CLOCKS_PER_SEC);
	begin = clock();
	getHome();
	end = clock();
	printf("Getting HOME path from env (optimized) took\t[%.06f]s\n", (double)(end - begin) / CLOCKS_PER_SEC);

	double time_default, time_opt, time_worse;
	for (int i = 0; i < 1; i++) {
		/*Scratch this : -Do a few runs to load in cache-*/
		long out;
		begin = clock();
		out = unlikelyLastout(iter, randomInt);
		end = clock();
		time_worse = (double)(end - begin) / CLOCKS_PER_SEC;

		begin = clock();
		out = normLastout(iter, randomInt);
		end = clock();
		time_default = (double)(end - begin) / CLOCKS_PER_SEC;

		begin = clock();
		out = likelyLastout(iter, randomInt);
		end = clock();
		time_opt = (double)(end - begin) / CLOCKS_PER_SEC;
	}

	printf("Worsened iteration took\t\t\t\t[%.06f]s\n", time_worse);
	printf("Default recursive iteration took\t\t[%.06f]s\n", time_default);
	printf("Optimized recursive iteration took\t\t[%.06f]s\n", time_opt);
	
	return 0;
}
