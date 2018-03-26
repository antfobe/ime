LIBS = -lpthread -lm
CC = gcc
CFLAGS = -O0 -Wall -std=c99
VARIABLES = -DPOND_SIZE=7

default: run

run: ep2
	./epm2

all: ep2

ep2: epm2.c
	$(CC) -o epm2 epm2.c $(CFLAGS) $(LIBS) $(VARIABLES)

clean:
	-rm -f epm2

