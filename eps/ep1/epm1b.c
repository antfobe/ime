/*
 * COMPILE gcc -Wall -O3 -sdc=c99
 */

#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SIZE (1024*1024*1024)
#define MAX_STEP (16*1024)

#pragma GCC diagnostic ignored "-Wunused-value"
#pragma GCC diagnostic ignored "-Wint-conversion"
int main()
{
    clock_t start, end;
    double cpu_time;
    int i = 0;
    int steps = MAX_STEP;

    /* MAX_SIZE array is too big for stack.This is an unfortunate rough edge of the way the stack works.
       It lives in a fixed-size buffer, set by the program executable's configuration according to the
       operating system, but its actual size is seldom checked against the available space. */

    char *M = (char *)malloc(MAX_SIZE * sizeof(char));

    for(int k = 1; k <= 1024; k <<= 1){
        /* CPU clock ticks count start */
        start = clock();
        /* Loop 2 */
        for (i = 0; i < MAX_SIZE && steps > 0; i += k, steps--)
            M[i] += 3;
        /* CPU clock ticks count stop */
        end = clock();
	steps = MAX_STEP;
        cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("CPU time for loop 2 (k = %d)\t[%.6f]s\n", k, cpu_time);
    }

    return 0;
}
