#include "ep5.h"

int nthreads = 2; 
long m = 0, n = 0, p = 0;
double ** A, ** B, ** C;
struct timeval begin, end;

int main(int argc, char ** argv){
char mode[8];
pthread_t * threads;
	if(argc != 5){
		USAGE();
	}

	nthreads = sysconf(_SC_NPROCESSORS_ONLN); /* Playtime is ogre */

	switch(argv[1][0]){
		case 'o':
			strcpy(mode, "OpenMP");
			if(argv[2]) A = readM(argv[2]);
			if(argv[3]) B = readM(argv[3]);
			C = alloc_initM(m, n); 

			gettimeofday(&begin, NULL);
			omp_mmul(m, n, p, A, B, C);
			gettimeofday(&end, NULL);
			
			if(!writeM(argv[4], C)) {
				DBG("failed to write C matrix to file"); exit(-1);
			}
		break;	
		case 'p':
			strcpy(mode, "Pthread");
			if(argv[2]) A = readM(argv[2]);
			if(argv[3]) B = readM(argv[3]);
			C = alloc_initM(m, n); 
			threads = (pthread_t *) malloc(nthreads * sizeof(pthread_t));
		
#ifdef DEBUG
	DBG("Start thread spawning");
#endif
			gettimeofday(&begin, NULL);
			for (int i = 0; i < nthreads; i++) {
				int * tid;
				tid = (int *) malloc(sizeof(int));
				* tid = i;
				pthread_create(&threads[i], NULL, pth_mworker, (void *)tid);
			}
			for (int j = 0; j < nthreads; j++) {
				pthread_join(threads[j], NULL);
			}
			gettimeofday(&end, NULL);

			free(threads);
			if(!writeM(argv[4], C)) {
				DBG("failed to write C matrix to file"); exit(-1);
			}
		break;	
		case 'O':
			strcpy(mode, "OpenMP");
			m = (int) strtol(argv[2], (char **)NULL, 10); if(!m){ USAGE();}
			n = (int) strtol(argv[3], (char **)NULL, 10); if(!n){ USAGE();}
			p = (int) strtol(argv[4], (char **)NULL, 10); if(!p){ USAGE();}

			A = alloc_onlyM(m, p);
			B = alloc_onlyM(p, n);
			C = alloc_onlyM(m, n);

			gettimeofday(&begin, NULL);
			omp_mmul(m, n, p, A, B, C);
			gettimeofday(&end, NULL);


		break;	
		case 'P':
			strcpy(mode, "Pthread");
			m = (int) strtol(argv[2], (char **)NULL, 10); if(!m){ USAGE();}
			n = (int) strtol(argv[3], (char **)NULL, 10); if(!n){ USAGE();}
			p = (int) strtol(argv[4], (char **)NULL, 10); if(!p){ USAGE();}

			A = alloc_onlyM(m, p);
			B = alloc_onlyM(p, n);
			C = alloc_onlyM(m, n);
			threads = (pthread_t *) malloc(nthreads * sizeof(pthread_t));

#ifdef DEBUG
	DBG("Start thread spawning");
#endif
			gettimeofday(&begin, NULL);
			for (int i = 0; i < nthreads; i++) {
				int * tid;
				tid = (int *) malloc(sizeof(int));
				* tid = i;
				pthread_create(&threads[i], NULL, pth_mworker, (void *)tid);
			}
			for (int j = 0; j < nthreads; j++) {
				pthread_join(threads[j], NULL);
			}
			gettimeofday(&end, NULL);

			free(threads);
			if(!writeM(argv[4], C)) {
				DBG("failed to write C matrix to file"); exit(-1);
			}
		break;
		default:
			USAGE();	
	}
#ifdef DEBUG
	DBG("Reached cleanup");
#endif
	freeM(m, A);
	freeM(p, B);
	freeM(m, C);
	tvsub(&end, &end, &begin);
	printf("%s matrix multiplication took [%ld.%06ld] seconds\n", mode, end.tv_sec, end.tv_usec);
	return 0;
}
