#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define POND_SIZE 7
#define DEADLOCK 9001

struct params {
        pthread_mutex_t mutex;
        pthread_cond_t done;
	char frogger[128];
        int id;
};

typedef struct params params_t;

int dead_count = 0;
int pond_pos[POND_SIZE];

/* Function to search a specific value pos in 
 * an array, returns -1 if pos is not found
 * or abs(pos) if value pos is within array */

int checkPos(int * array, int pos) {
	for(int i = 0; i < sizeof(array)/sizeof(array[0]); i++) {
		if(pos == array[i])
			return abs(pos);
	}
	return -1;
}

void * pond(void * arg){

    int id;
    char * frogger;
    int i;
    /* Lock.  */
    while(dead_count < DEADLOCK) {
	    pthread_mutex_lock(&(*(params_t*)(arg)).mutex);

	    /* Work.  */
	    id = (*(params_t*)(arg)).id;
	    frogger = (*(params_t*)(arg)).frogger;
	    
	    printf("State: ");
	    for(i = 0; i < (POND_SIZE); i++) {
		printf("[%d]{%d} ", i, pond_pos[i]);
	    }
	    printf(" DeadCount=%d\n", dead_count);
	    
	    dead_count++;

	    //printf("Position[%d]: Check[+1]={%d},Check[+2]{%d}; Check[-1]={%d},Check[-2]{%d}\n", pond_pos[id], checkPos(pond_pos, pond_pos[id] + 1), checkPos(pond_pos, pond_pos[id] + 2), checkPos(pond_pos, pond_pos[id] - 1), checkPos(pond_pos, pond_pos[id] - 2));

	    /* Logic is as follows: if male frogger, move forwards, else backwards;
	     * - priority will be giver to simple movement[+1|-1] rather than jump[+2|-2]
	     * - if a move was made, reset dead_count;  */

	    if (!strstr(frogger, "Female")) {
		if (pond_pos[id] < POND_SIZE && checkPos(pond_pos, pond_pos[id] + 1) ) {
			pond_pos[id]++;
			dead_count = 0;
		} else if (pond_pos[id] < (POND_SIZE - 1) && checkPos(pond_pos, pond_pos[id] + 2)) {
			pond_pos[id] += 2;
			dead_count = 0;
		}

	    } else {
		if (pond_pos[id] > 1 && checkPos(pond_pos, pond_pos[id] - 1) ) {
			pond_pos[id]--;
			dead_count = 0;
		} else if (pond_pos[id] > 2 && checkPos(pond_pos, pond_pos[id] - 2)) {
			pond_pos[id] -= 2;
			dead_count = 0;
		}
	    }

	    /* Unlock and signal completion.  */
	    pthread_mutex_unlock(&(*(params_t*)(arg)).mutex);
	    pthread_cond_signal (&(*(params_t*)(arg)).done);
    }
    /* After signalling `main`, the thread could actually
     * go on to do more work in parallel. Thought lets not -
     * returning NULL to finish thread  */

    return NULL;
}


int main() {

    if(POND_SIZE % 2 == 0) {
	printf("ERROR - POND_SIZE must be an odd number, exiting\n");
	exit(EXIT_FAILURE);
    }

    pthread_t threads[POND_SIZE - 1];
    params_t params;
    pthread_mutex_init (&params.mutex , NULL);
    pthread_cond_init (&params.done, NULL);

    /* Obtain a lock on the parameter.  */
    pthread_mutex_lock (&params.mutex);

    /* Initialize stuff for pond problem. */
    int i;
    pond_pos[POND_SIZE - 1] = 0;
    for(i = 0; i < (POND_SIZE - 1); i++) {
	 if ( i % 2 ) {
	    	pond_pos[i] = (i + POND_SIZE)/2 + 1;
	    } else {
	    	pond_pos[i] = (i)/2 + 1;
	    }
    }

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
            pthread_cond_wait (&params.done, &params.mutex);
    }

    //for(i = 0; i < (POND_SIZE - 1); i++) { pthread_join(threads[i], NULL); }

    /* Destroy all synchronization primitives.  */    
    pthread_mutex_destroy (&params.mutex);
    pthread_cond_destroy (&params.done);

    printf("End state:\n");
    for(i = 0; i < (POND_SIZE - 1); i++) {
	if (i%2) {
		printf("Female frog id[%d],\tPosition{%d}\n", i, pond_pos[i]);
	} else {
		printf("Male frog   id[%d],\tPosition{%d}\n", i, pond_pos[i]);
	}
    }
    printf("Deadcount [%d]\n", dead_count);
    return (0);
}
