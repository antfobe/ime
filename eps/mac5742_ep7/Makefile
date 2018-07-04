CC = nvcc
CFLAGS = -c -O3  -arch sm_30  -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi
LFLAGS = -O2 -w -D Tile_Width=128 -arch sm_30 -Xcompiler -fopenmp -lgomp -lmpi_cxx -lmpi -lopen-rte -lopen-pal -ldl -lnsl -lutil -lm -L/usr/lib/openmpi/
AUTHOR = jean.fobe.eugenio.gimenes

default: all

all: ep7

ep7: ep7.cu
	$(CC) $(CFLAGS) ep7.cu
	$(CC) $(LFLAGS) ep7.o -o ep7

test: all 
	-mpirun -np 1 ep7
	-mpirun -np 2 ep7 $$((1024*1024)) 1024 8192

zip:
	-zip -uT9v $(AUTHOR).zip Makefile ep7* *.md *.txt

clean:
	rm -rf *.o ep7 *.zip
