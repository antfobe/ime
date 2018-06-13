CC = nvcc
CFLAGS = -lm -O2 -w -D Tile_Width=128 -arch sm_35
AUTHOR = jean.fobe.eugenio.gimenes

default: all

all: ep6

ep6: ep6.cu
	$(CC) -o ep6 $(CFLAGS) ep6.cu

test: all 
	./ep6 sample.txt

zip:
	-zip -uT9v $(AUTHOR).zip Makefile ep6* *.md *.txt

clean:
	rm -rf *.o ep6
