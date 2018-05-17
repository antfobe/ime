LIBS = -fopenmp -lpthread
CC = gcc
CFLAGS = -O0 -Wall
DEFAULTS = 1024 512 2048
AUTHOR = jean.fobe

default: run

debug: ep5.c ep5.h ep5_main.c
	$(CC) -o ep5 ep5.c ep5_main.c $(CFLAGS) -DDEBUG $(LIBS)

bench: ep5
	-time for m in $$(seq 1 12) ; do for p in $$(seq 1 16) ; do for n in $$(seq 1 12) ; do echo -n "OMP: m=$$((2**$$m)) p=$$((2**$$p)) n=$$((2**$$n)) - " && ./ep5 O $$((2**$$m)) $$((2**$$p)) $$((2**$$n)); done done done
	-time for m in $$(seq 1 12) ; do for p in $$(seq 1 16) ; do for n in $$(seq 1 12) ; do echo -n "PTH: m=$$((2**$$m)) p=$$((2**$$p)) n=$$((2**$$n)) - " && ./ep5 P $$((2**$$m)) $$((2**$$p)) $$((2**$$n)); done done done

run: ep5
	./ep5 O $(DEFAULTS)
	./ep5 P $(DEFAULTS)

all: ep5

ep5: ep5.c ep5.h ep5_main.c
	$(CC) -o ep5 ep5.c ep5_main.c $(CFLAGS) $(LIBS)

zip:
	-zip -uT9v $(AUTHOR).zip Makefile *.c *.h *.md *.txt

clean:
	-rm -f ep5
	-rm -f ep5d
