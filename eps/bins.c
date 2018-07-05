#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <sys/random.h>
#include <sys/syscall.h>

#define _GNU_SOURCE
#define USAGE() do { printf("Usage: ./bins { number, number, ..., number }\n"); exit(0); } while(0)
    
void showbits(unsigned int x){
    int i; 
    for(i=(sizeof(int)*8)-1; i>=0; i--)
            (x&(1u<<i))?putchar('1'):putchar('0');
    
    printf("\n");
}

void shortbits(short x){
    int i; 
    for(i=(sizeof(short)*8)-1; i>=0; i--)
            (x&(1u<<i))?putchar('1'):putchar('0');
    
    printf("\n");
}

int main(int argc, char * argv[]) {
	srand(time(NULL));
	unsigned int j = 0;
	for(int args = 1; args < argc; args++) {
		j = (unsigned int) strtol(argv[args], (char **)NULL, 10); if(!j){ USAGE();}
		printf("%u in binary\t\t", j);
		/* assume we have a function that prints a binary string when given 
		   a decimal integer 
		 */
		showbits(j);
	}
	/*int m, n;
	// the loop for right shift operation 
	for ( m = 0; m <= 5; m++ ) {
	    n = j >> m;
	    printf("%d right shift %d gives \t", j, m);
	    showbits(n);
	}
	*/
	syscall(SYS_getrandom, &j, sizeof(unsigned int), 0);
	unsigned short n = j>>16;
	unsigned short m = j;
	shortbits(n);
	shortbits(m);
	shortbits(n ^ m);
	n = n ^ m; /* XORn */
	printf("XORn = %d : \t\t\t", n); shortbits(n);
	for (unsigned short p = 0, find = 0, exit = 0; p < USHRT_MAX && exit == 0; p++){
		/* found m candidate */
		if (m == n ^ p) {
			printf("found n! (find,p) = (%d,%d)\n", n, p);
			shortbits(find);
			exit = 1;
		}
	}
	//m = 1 - n;
	//n = n ^ m; /* Lost n, now XORn (XORn = n ^ m) */
	//m = m ^ n; /* Lost m, now XORm (XORm = XORn ^ m)*/
	//n = n ^ m; /* recover m on n (n = XORn ^ XORm)*/
	//m = m ^ n; /* recover n on m (m = XORm ^ n)*/
	//showbits(n);
	//showbits(m);
	return 0;
}

