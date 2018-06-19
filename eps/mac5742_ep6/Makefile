CC = nvcc
CFLAGS = -lm -O2 -w -D Tile_Width=128 -arch sm_35
AUTHOR = jean.fobe.eugenio.gimenes

default: all

sample: gensample.sh
	./gensample.sh $$(( (RANDOM % 115) )) sample.txt

all: ep6

ep6: ep6.cu
	$(CC) -o main $(CFLAGS) ep6.cu

test: all sample
	./main sample.txt

zip:
	-zip -uT9v $(AUTHOR).zip Makefile ep6* *.md *.txt

clean:
	rm -rf *.o ep6 main
