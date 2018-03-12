LIBS = 
CC = gcc
CFLAGS = -O0 -Wall -std=c99

default: run

run: ep1a ep1b
	time ./epm1a &&	time ./epm1b

all: ep1a ep1b

runa: ep1a
	./epm1a

runb: ep1b
	./epm1b

ep1a: epm1a.c
	$(CC) $(CFLAGS) $(LIBS) -o epm1a epm1a.c

ep1b: epm1b.c
	$(CC) $(CFLAGS) $(LIBS) -o epm1b epm1b.c

clean:
	-rm -f epm1a
	-rm -f epm1b
