LIBS = -lpthread -lm
CC = gcc
CFLAGS = -O0 -Wall
VARIABLES = -DPOND_SIZE=7

default: run

run: ep2
	./epm2

all: ep2

ep2: epm2.c
	$(CC) -o epm2 epm2.c $(CFLAGS) $(LIBS) $(VARIABLES)

debug:
	$(CC) -o epm2 epm2.c $(CFLAGS) $(LIBS) $(VARIABLES) -DDEBUG

clean:
	-rm -f epm2
