CC = nvcc
CFLAGS = -lm -O2 -w -D Tile_Width=128 -arch sm_35
AUTHOR = jean.fobe.eugenio.gimenes

default: all

all: ep7

ep7: ep7.cu
	$(CC) -c -O3  -arch sm_30  -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi ep7.cu
	$(CC) -lm  -lcudart -lcublas  -L/usr/lib/openmpi/ -lmpi_cxx -lmpi -lopen-rte -lopen-pal -ldl -lnsl -lutil -lm ep7.o -o ep7

test: all 
	-mpirun -np 2 ep7 65536 8192 1024

zip:
	-zip -uT9v $(AUTHOR).zip Makefile ep7* *.md *.txt

clean:
	rm -rf *.o ep7 *.zip
