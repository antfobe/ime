LIBS = -lpthread
CC = gcc
CFLAGS = -O0 -Wall -std=c99

default: run

run: ep2
	./epm2

all: ep2

ep2: epm2.c
	$(CC) $(CFLAGS) $(LIBS) -o epm2 epm2.c

clean:
	-rm -f epm2

