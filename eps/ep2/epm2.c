#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define POND_SIZE 7

struct params {
        pthread_mutex_t mutex;
        pthread_cond_t done;
	char frogger[128];
	int pond_pos[POND_SIZE];
        int id;
};

typedef struct params params_t;

int dead_count = 0;

int checkPos(int * array, int pos) {
	for(int i = 0; i < sizeof(array)/sizeof(array[0]); i++) {
		if(pos == array[i])
			return abs(pos);
	}
	return -1;
}

void * pond(void * arg){

    int id, * pos;
    /* Lock.  */
    pthread_mutex_lock(&(*(params_t*)(arg)).mutex);

    /* Work.  */
    id = (*(params_t*)(arg)).id;
    pos = (*(params_t*)(arg)).pond_pos;
    
    printf("Hello from %d - {%s}\n", id, (*(params_t*)(arg)).frogger);
    
    /* Try to move. */
    if (id <= POND_SIZE/2) {
	if (pos[id] < (POND_SIZE - 1) && ! checkPos(pos, pos[id] + 1) && checkPos(pos, pos[id] + 2)) {
		pos[id]++;
	} else if (pos[id] < (POND_SIZE - 1) && ! checkPos(pos, pos[id] + 2)) {
		pos[id] += 2;
	}

    } else {
	if (pos[id] > 0 && ! checkPos(pos, pos[id] - 1) && checkPos(pos, pos[id] - 2)) {
		pos[id]--;
	} else if (pos[id] > 0 && ! checkPos(pos, pos[id] - 2)) {
		pos[id] -= 2;
	}
    }

  /* Check if moved...  */

    if (pos[id] == (*(params_t*)(arg)).pond_pos[id]) {
    	dead_count++;
    } else {
	(*(params_t*)(arg)).pond_pos[id] = pos[id];
	dead_count = 0;
    }

    /* Unlock and signal completion.  */
    pthread_mutex_unlock(&(*(params_t*)(arg)).mutex);
    pthread_cond_signal (&(*(params_t*)(arg)).done);

    /* After signalling `main`, the thread could actually
    go on to do more work in parallel.  */
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
    params.pond_pos[POND_SIZE - 1] = 0;
    for(i = 0; i < (POND_SIZE - 1); i++) {

            /* Change the parameter (I own it).  */    
            params.id = i;
            if ( i % 2 ) {
	    	sprintf(params.frogger, "Female frogger [%d]", (i) / 2);
	    	params.pond_pos[i] = (i + POND_SIZE)/2 + 1;
	    } else {
	    	sprintf(params.frogger, "Male   frogger [%d]", (i + 1) / 2);
	    	params.pond_pos[i] = (i)/2 + 1;
	    }
            
	    /* Spawn a thread.  */
            pthread_create(&threads[i], NULL, pond, &params);

            /* Give up the lock, wait till thread is 'done',
            then reacquire the lock.  */
            pthread_cond_wait (&params.done, &params.mutex);
    }

    for(i = 0; i < (POND_SIZE - 1); i++) {
            pthread_join(threads[i], NULL);
    }

    /* Destroy all synchronization primitives.  */    
    pthread_mutex_destroy (&params.mutex);
    pthread_cond_destroy (&params.done);

    for(i = 0; i < (POND_SIZE); i++) {
	printf("UWOT[%d] = {%d}\n", i, params.pond_pos[i]);
    }
	printf("  X.x -- Deadcount <%d> -- x.X\n", dead_count);
    return (0);
}
