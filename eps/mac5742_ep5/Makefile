OMP = -fopenmp 
PTH = -lpthread
CC = gcc
CFLAGS = -O0 -Wall
DEFAULTS = 1024 512 2048

default: run

bench: omp pth
	-time for m in $$(seq 1 12) ; do for p in $$(seq 1 16) ; do for n in $$(seq 1 12) ; do echo -n "OMP: m=$$((2**$$m)) p=$$((2**$$p)) n=$$((2**$$n)) - " && ./omp_ep5 $$((2**$$m)) $$((2**$$p)) $$((2**$$n)); done done done
	-time for m in $$(seq 1 12) ; do for p in $$(seq 1 16) ; do for n in $$(seq 1 12) ; do echo -n "PTH: m=$$((2**$$m)) p=$$((2**$$p)) n=$$((2**$$n)) - " && ./pth_ep5 $$((2**$$m)) $$((2**$$p)) $$((2**$$n)); done done done

run: omp pth
	./omp_ep5 $(DEFAULTS)
	./pth_ep5 $(DEFAULTS)

all: omp pth

omp: omp_ep5.c
	$(CC) -o omp_ep5 omp_ep5.c $(CFLAGS) $(OMP)

pth: pth_ep5.c
	$(CC) -o pth_ep5 pth_ep5.c $(CFLAGS) $(PTH)

clean:
	-rm -f omp_ep5
	-rm -f pth_ep5
