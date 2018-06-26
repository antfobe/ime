CC = nvcc
CFLAGS = -lm -O2 -w -D Tile_Width=128 -arch sm_35
AUTHOR = jean.fobe.eugenio.gimenes

default: all

sample: gensample.sh
	./gensample.sh $$(( (RANDOM % 115) )) sample.txt

all: ep7

ep7: ep7.cu
	module load openmpi cuda #(optional) load modules on your node
	mpicc -c main.c -o main.o
	nvcc -arch=sm_20 -c multiply.cu -o multiply.o
	mpicc main.o multiply.o -lcudart -L/apps/CUDA/cuda-5.0/lib64/ -o program

test: all sample
	./main sample.txt

zip:
	-zip -uT9v $(AUTHOR).zip Makefile ep6* *.md *.txt

mc: montecarlo.c
	gcc -o mc -lm montecarlo.c

clean:
	rm -rf *.o ep6 main mc
