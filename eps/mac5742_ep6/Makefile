CC=nvcc
CFLAGS=-lm -O2 -w -D Tile_Width=128 -arch sm_35
DEVICE_ID=0
sizeV=262144
sizeM=2048 

all: matrix_sum_normal matrix_sum_coalesced matrix_sum_cpu

matrix_sum_cpu: 
	$(CC) $(CFLAGS) matrix_sum_cpu.cu -o matrix_sum_cpu

matrix_sum_normal.o: matrix_sum_normal.cu
	$(CC) $(CFLAGS) -c matrix_sum_normal.cu

matrix_sum_normal: matrix_sum_normal.o
	$(CC) $(CFLAGS) matrix_sum_normal.o -o matrix_sum_normal

matrix_sum_coalesced.o: matrix_sum_coalesced.cu
	$(CC) $(CFLAGS) -c matrix_sum_coalesced.cu

matrix_sum_coalesced: matrix_sum_coalesced.o 
	$(CC) $(CFLAGS) matrix_sum_coalesced.o -o matrix_sum_coalesced

tests: all 
	./matrix_sum_normal $(sizeM) $(DEVICE_ID)
	./matrix_sum_coalesced $(sizeM) $(DEVICE_ID) 
	./matrix_sum_cpu $(sizeM)

clean:
	rm -rf *.o matrix_sum_normal matrix_sum_coalesced matrix_sum_cpu
