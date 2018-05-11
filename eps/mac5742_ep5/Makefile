OMP = -fopenmp 
PTH = -lpthread
CC = gcc
CFLAGS = -O0 -Wall
DEFAULTS = 1024 512 2048

default: run

run: omp
	./omp_ep5 $(DEFAULTS)

all: omp pth

omp: omp_ep5.c
	$(CC) -o omp_ep5 omp_ep5.c $(CFLAGS) $(OMP)

pth: pth_ep5.c
	$(CC) -o pth_ep5 pth_ep5.c $(CFLAGS) $(PTH)

clean:
	-rm -f omp_ep5
	-rm -f pth_ep5
