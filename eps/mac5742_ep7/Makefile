CC = nvcc
CFLAGS = -lm -O2 -w -D Tile_Width=128 -arch sm_35
AUTHOR = jean.fobe.eugenio.gimenes

default: all

all: ep7

ep7: ep7.cu
	$(CC) -c -O3  -arch sm_30  -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi ep7.cu
	$(CC) -lm  -lcudart -lcublas  -L/usr/lib/openmpi/ -lmpi_cxx -lmpi -lopen-rte -lopen-pal -ldl -lnsl -lutil -lm ep7.o -o ep7

mc: montecarlo.cu
	$(CC) -c -O3  -arch sm_30  -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi montecarlo.cu
	$(CC) -lm  -lcudart -lcublas  -L/usr/lib/openmpi/ -lmpi_cxx -lmpi -lopen-rte -lopen-pal -ldl -lnsl -lutil -lm montecarlo.o -o mc

test: all 
	-mpirun -np 2 ep7 65536 8192 1024

zip:
	-zip -uT9v $(AUTHOR).zip Makefile ep7* *.md *.txt

clean:
	rm -rf *.o ep7


##OMPI_CXX=nvcc OMPI_CXXFLAGS="-I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi" OMPI_LDFLAGS="-L/usr/lib/openmpi/lib -lmpi_cxx -lmpi -lopen-rte -lopen-pal -ldl -lnsl -lutil -lm -ld" OMPI_LIBS="" mpicxx -c -O3 -arch sm_30 *.cu && mpicxx -L/opt/cuda/lib64 -lcudart -lcublas *.o -o main.NVCC_MPI_
