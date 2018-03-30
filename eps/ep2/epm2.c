#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef POND_SIZE
	#define POND_SIZE 7
#endif

#ifndef MAIN_RATE
	#define MAIN_RATE 500000000L
#endif

#define DEADLOCK 32 
#define MAX_TRIES 256
#define t1ms 1000000L

struct params {
	pthread_cond_t done;
	char frogger[128];
        int id;
};

typedef struct params params_t;

pthread_mutex_t mutex;

int dead_count = 0;
int pond_pos[POND_SIZE];

/* Function to check a specific value pos in 
 * an array, returns 0 if pos is not found
 * or 1 if value pos is within array */

int checkPos(int * array, int pos) {
	for(int i = 0; i < sizeof(array)/sizeof(array[0]); i++) {
		if(pos == array[i])
			return 1;
	}
	return 0;
}

void * pond(void * arg){

    int id;
    char * frogger;
    
    /* Thread infinite loop, threads will try to update pond_pos array, 
     * if a deadlock is reached dead_count will be incremented 
     * non-stop */

    /* Work.  */
    id = (*(params_t*)(arg)).id;
    frogger = (*(params_t*)(arg)).frogger;
	    
    /* Signal done to main. */
    pthread_cond_signal (&(*(params_t*)(arg)).done);

    /* After signalling `main`, the thread will go on to do more work 
     * in parallel. Giving it a quick nap */

    //nanosleep((const struct timespec[]){{0, 1000 * t1ms}}, NULL);
	    
    while(1) {

	    /* Sleep a bit. */
	    nanosleep((const struct timespec[]){{0, id * t1ms}}, NULL);
	    
    	    /* Lock.  */
	    pthread_mutex_lock(&mutex);

	    dead_count++;

	    /* Logic is as follows: if male frogger, move forwards, else backwards;
	     * - priority will be given to simple movement[+1|-1] rather than jump[+2|-2]
	     * - if a move was made, reset dead_count;  */

	    if (!strstr(frogger, "Female")) {
		if (pond_pos[id] < POND_SIZE && !checkPos(pond_pos, pond_pos[id] + 1) ) {
			/*printf("%s Moved: [%d]->[%d]\n", frogger, pond_pos[id], pond_pos[id] + 1);*/
			pond_pos[id] = pond_pos[id] + 1;
			dead_count = 0;
		} else if (pond_pos[id] < (POND_SIZE - 1) && !checkPos(pond_pos, pond_pos[id] + 2)) {
			pond_pos[id] = pond_pos[id] + 2;
			dead_count = 0;
		}
	    } else {
		if (pond_pos[id] > 1 && !checkPos(pond_pos, pond_pos[id] - 1) ) {
			pond_pos[id] = pond_pos[id] - 1;
			dead_count = 0;
		} else if (pond_pos[id] > 2 && !checkPos(pond_pos, pond_pos[id] - 2)) {
			pond_pos[id] = pond_pos[id] - 2;
			dead_count = 0;
		}
	    }

	    /* Unlock.  */
	    pthread_mutex_unlock(&mutex);
    }

    return NULL;
}


int main(int argc, char ** argv) {
    
    if(POND_SIZE % 2 == 0) {
	printf("ERROR - POND_SIZE must be an odd number, exiting\n");
	exit(EXIT_FAILURE);
    }

    pthread_t threads[POND_SIZE - 1];
    params_t params;
    pthread_mutex_init (&mutex , NULL);
    pthread_cond_init (&params.done, NULL);

    /* Obtain a lock on the parameter.  */
    pthread_mutex_lock (&mutex);

    /* Initialize stuff for pond problem. */
    int i, tries = 0;
    srand(time(NULL));
    int solution[POND_SIZE];
    pond_pos[POND_SIZE - 1] = 0;
    for(i = 0; i < (POND_SIZE - 1); i++) {
	 if ( i % 2 ) {
	    	pond_pos[i] = (i + POND_SIZE)/2 + 1;
	    } else {
	    	pond_pos[i] = (i)/2 + 1;
	    }
    }

#ifdef DEBUG
	printf("Starting postions: \n");
    	for(i = 0; i < (POND_SIZE - 1); i++) { printf("[%d]{%d} ", i, pond_pos[i]);}	
	printf("\n");
#endif

    for(i = 0; i < (POND_SIZE - 1); i++) {

            /* Change the parameter (I own it).  */    
            params.id = i;
            if ( i % 2 ) {
	    	sprintf(params.frogger, "Female frogger [%d]", (i) / 2);
	    } else {
	    	sprintf(params.frogger, "Male   frogger [%d]", (i + 1) / 2);
	    }
            
	    /* Spawn a thread.  */
            pthread_create(&threads[i], NULL, pond, &params);

            /* Give up the lock, wait till thread is 'done',
            then reacquire the lock.  */
            pthread_cond_wait (&params.done, &mutex);
    }
    	/* Unlock... */
	//for(i = 0; i < POND_SIZE - 2; i++) { pthread_join(threads[i], NULL); }
	pthread_mutex_unlock(&mutex);

    /* Loop until acceptable solution. */
    int working = 1;
    while(working && tries <= MAX_TRIES) {
	
	/*Sleep a bit... */
	nanosleep((const struct timespec[]){ {0, MAIN_RATE} }, NULL);

	/* Lock. */
	pthread_mutex_lock(&mutex);

	/** DEBUGGING **/
#ifdef DEBUG
    	for(i = 0; i < (POND_SIZE - 1); i++) {
		printf("[%d]{%d} ", i, pond_pos[i]);
	}	printf(" -- DEADCOUNT [%d]/TRY [%d]\n", dead_count, tries);
#endif 

	/* Check if solution, else reset problem (frog postions) */
	if(dead_count >= DEADLOCK) {
    		for(i = 0; i < (POND_SIZE - 1); i++) {
			if (i%2) {
				working += (pond_pos[i] < (POND_SIZE + 1) / 2);
			} else {
				working += (pond_pos[i] > (POND_SIZE + 1) / 2);
			}
		}
		if((double)(rand() % MAX_TRIES) > (double)(0.75 * MAX_TRIES)) {

			/* If at first not deterministic random 
			 * behavior, force determinism. */
			working = 0;	
	    		for(i = 0; i < (POND_SIZE - 1); i++) {
				if( i % 2 ) { 
					solution[i] = (i)/2 + 1;
				} else { 
					solution[i] = (i + POND_SIZE)/2 + 1;
				}
			}

		} else if(working > 1) { 

			/* Reset pond_pos. */
			pond_pos[POND_SIZE - 1] = 0;
			for(i = 0; i < (POND_SIZE - 1); i++) {
			     if ( i % 2 ) {
				pond_pos[i] = (i + POND_SIZE)/2 + 1;
				} else {
				pond_pos[i] = (i)/2 + 1;
				}
			}

			/* Increment incorrect solution count */
			tries++;

		} else {
			
			/* Arrived at solution, exit loop. */
			working = 0;

			/* Get solution, just in case... */
			memcpy(solution, pond_pos, sizeof(pond_pos));
		}

	/* Reset dead_count. */
	dead_count = 0;

	}
	
	/* & Unlock. */
	pthread_mutex_unlock(&mutex);
    }

    /* Destroy all synchronization primitives.  */    
    pthread_mutex_destroy (&mutex);
    pthread_cond_destroy (&params.done);

    printf("End state:\n");
    if (tries >= MAX_TRIES) {
	    printf("NO SOLUTION REACHED - STOPPING AFTER [%d] TRIES\n", tries - 1);
    } else {
	    for(i = 0; i < (POND_SIZE - 1); i++) {
		if (i%2) {
			printf("Female frog id[%d],\tPosition{%d}\n", i, solution[i]);
		} else {
			printf("Male frog   id[%d],\tPosition{%d}\n", i, solution[i]);
		}
	    }
	    printf("\nTotal amount of tries: [%d]\n", tries);
    }
    return (0);
}
