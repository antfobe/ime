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
#define MALLOC_CHECK(x) do { printf("\nMemmory allocation error at {%s}, line [%d]\n", x, __LINE__); exit(EXIT_FAILURE); } while(0)

void sha256(char *string, unsigned char outputBuffer[65]){
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

unsigned char xorstring_reduce(unsigned char * string){
    for(int i = 1; string[i] != '\0' && strlen(string) > 1; i++){
        string[0] ^= string[i];
    }
    return string[0];
}

unsigned char * xor_expand(unsigned char seed, unsigned char * stophash, unsigned long stopsize) {
    short pow = 1, carry = 0;
    unsigned char * output = NULL, * outxor = NULL, hash[65] = {};
        
    while(strcmp(hash, stophash) != 0 && pow <= stopsize) {
        outxor = realloc(outxor, pow * sizeof(char));
        if(outxor == NULL) {MALLOC_CHECK("outxor");}
        if(pow == 1) {outxor[0] = seed;}
            
        pow <<= 1;
        output = realloc(output, pow * sizeof(char));
        if(output == NULL) {MALLOC_CHECK("output");}

        for(long i = 0; i < pow/2 && pow > 2; i++){
            outxor[i] = output[i];
        }

            //printf("output \t<%s>\n", output);
        for(long i = 0, j = 0; i < pow/2; i++) {
            //for(long j = 0; j <= i; j++) {
            while(carry && j < pow/2){
                if(output[j] == 255){
                    output[j] = 0;
                    j++;
                } else {
                    output[j]++;
                    carry = 0;
                    j = 0;
                }
            }
            for(output[0] = 0; output[0] < UCHAR_MAX; output[0]++){
                for(long x = 0; x < pow/2; x++) {
                    output[x+pow/2] = output[x] ^ outxor[x];
                }
                sha256(output, hash);
                if(strcmp(hash, stophash) == 0) {
                    free(outxor);
                    return output;
                }
            }
            if(pow >= 4) printf("o[0-3] = %d, %d, %d, %d;\n", output[0],output[1],output[2],output[3]);
            carry++;
        }
    }
    return NULL;
}

unsigned char * read4file(char * filename, long length){
    FILE * f = fopen(filename, "r");
    char * buffer = NULL;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fsize < length) {
        length = fsize;
    }
    buffer = (char *)malloc(fsize * sizeof(char));
    if(buffer == NULL) {MALLOC_CHECK("buffer");}
    
    char format[22] = {};
    sprintf(format, "%%%ds", length);
    fscanf(f, format, buffer);

    //printf("buffer is :%s\n", buffer);

    fclose(f);

    return buffer;
}

void showbits(unsigned int x, char * buffer){
    strcpy(buffer, "");
    for(int i = (sizeof(int)*8)-1; i >= 0; i--)
            (x&(1u<<i))?strncat(buffer, "1", 1):strncat(buffer, "0", 1);
}

void shortbits(short x){
    int i; 
    for(i=(sizeof(short)*8)-1; i>=0; i--)
            (x&(1u<<i))?putchar('1'):putchar('0');
    
    printf("\n");
}

int main(int argc, char * argv[]) {
	srand(time(NULL));
	unsigned int j = '0';
    char bits[33];
    unsigned char hash[65];
	for(int args = 1; args < argc; args++) {
		j = (unsigned int) strtol(argv[args], (char **)NULL, 10); if(!j){ USAGE();}
		printf("%u in binary\t\t", j);
		/* assume we have a function that prints a binary string when given 
		   a decimal integer 
		 */
		showbits(j, bits);
        sha256(bits, hash);
        printf("bits -- %s\nsha256:\t%s\n", bits, hash);
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
    
    unsigned char check[65];
    showbits(j, bits);
    sha256(bits, hash);
        printf("bits -- %s\nsha256:\t%s\n", bits, hash);
	unsigned short m = j>>16;
	unsigned short n = j;
	printf("n = \t\t");shortbits(n);
	printf("m = \t\t");shortbits(m);
	printf("n = n ^ m,\t"); n = n ^ m; shortbits(n);//n = n ^ m; /* Lost n, now XORn (XORn = n ^ m) */
	printf("m = m ^ n,\t"); m = m ^ n; shortbits(m);//m = m ^ n; /* Lost m, recover n (n = XORn ^ m)*/
	printf("n = n ^ m,\t"); n = n ^ m; shortbits(n);//n = n ^ m; /* recover m on n (m = XORn ^ n)*/
	printf("m = \t\t"); shortbits(m);
    
    showbits(((unsigned int)n<<16) + (unsigned int)m, bits);
    sha256(bits, hash);
        printf("bits -- %s\nsha256:\t%s\n", bits, hash);
	for (unsigned short pm = 0, find = 0; pm < USHRT_MAX; pm++){
	/* needs binary search magick */
		/* found m candidate */
		find = pm ^ (n ^ m); // possible n value
        showbits(((unsigned int)pm<<16) + (unsigned int)find, bits);
        sha256(bits, check);
		if (strcmp(hash, check) == 0) {
			printf("found n! (find,p) = (%d,%d)\n", find, pm);
			printf("m: ");shortbits(pm);
			printf("n: ");shortbits(find);
            printf("bits -- %s\nsha256:\t%s\n\n", bits, check);
		}
	}
    
    unsigned char * encode = read4file("random.txt", 4);
    printf("encode[0-3]: %d, %d, %d, %d -- %s\n", encode[0], encode[1], encode[2], encode[3], encode);
    sha256(encode, hash);
    printf("sha256:\t%s\n", hash);
    //printf("sha256:\t%s\n", hash);
    
    unsigned char decode = xorstring_reduce(encode);
    unsigned char * decoded = xor_expand(decode, hash, 4);
    //unsigned char decoded[4];sprintf(decoded,"%c%c%c%c", 60,158,244,134 );//xor_expand(decode, hash);
    if(decoded) {
        printf("decoded[0-3]: %d, %d, %d, %d -- %s\n", decoded[0], decoded[1], decoded[2], decoded[3], decoded);
        sha256(decoded, hash);
        printf("sha256:\t%s\n", hash);
    }
	return 0;
}
