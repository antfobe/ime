/* COMPILE gcc -o bins -lcrypto bins.c */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <sys/random.h>
#include <sys/syscall.h>
#include <openssl/sha.h>

#define _GNU_SOURCE
#define USAGE() do { printf("Usage: ./bins { number, number, ..., number }\n"); exit(0); } while(0)
   
void sha256(char *string, char outputBuffer[65]){
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, string, strlen(string));
    SHA256_Final(hash, &sha256);
    for(int i = 0; i < SHA256_DIGEST_LENGTH; i++){
        sprintf(outputBuffer + (i * 2), "%02x", hash[i]);
    }
    outputBuffer[64] = 0;
}

char * showbits(unsigned int x, char * buffer){
    strcpy(buffer, "");
    for(int i = (sizeof(int)*8)-1; i >= 0; i--)
            (x&(1u<<i))?strncat(buffer, "1", 1):strncat(buffer, "0", 1);
    return buffer;    
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
    char bits[33];
	for(int args = 1; args < argc; args++) {
		j = (unsigned int) strtol(argv[args], (char **)NULL, 10); if(!j){ USAGE();}
		printf("%u in binary\t\t", j);
		/* assume we have a function that prints a binary string when given 
		   a decimal integer 
		 */
		showbits(j, bits);
        printf("bits - %s\n", bits);
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
	unsigned short m = j>>16;
	unsigned short n = j;
	printf("n = \t\t");shortbits(n);
	printf("m = \t\t");shortbits(m);
	printf("n = n ^ m,\t"); n = n ^ m; shortbits(n);//n = n ^ m; /* Lost n, now XORn (XORn = n ^ m) */
	printf("m = m ^ n,\t"); m = m ^ n; shortbits(m);//m = m ^ n; /* Lost m, recover n (n = XORn ^ m)*/
	printf("n = n ^ m,\t"); n = n ^ m; shortbits(n);//n = n ^ m; /* recover m on n (m = XORn ^ n)*/
	printf("m = \t\t"); shortbits(m);
	for (unsigned short pm = 0, find = 0; pm < USHRT_MAX; pm++){
	/* binary search magick
	for (unsigned short low = 0, mid, high = USHRT_MAX - 1, find; low <= high;){
		mid = (low + high) / 2;
		*/
		/* found m candidate */
		find = pm ^ (n ^ m); // possible n value
		if (n == find && m == find ^ pm ) {
			printf("found n! (find,p) = (%d,%d)\n", find, pm);
			printf("m: ");shortbits(pm);
			printf("n: ");shortbits(find);
		}
	}
	return 0;
}

