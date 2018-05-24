all: dotProd dotCpu

include common/Makefile.common

dotCpu: 
	$(CC) $(CFLAGS) -o dot_cpu dot_cpu.cu

dotProd.o: dotProd.cu
	$(CC) $(CFLAGS) -c dotProd.cu

dotProd: dotProd.o
	$(CC) $(CFLAGS) dotProd.o -o dotProd

tests: dotProd dotCpu
	./dotProd $(sizeV) $(DEVICE_ID) 
	./dot_cpu $(sizeV)


clean:
	rm -rf *.o dotProd dot_cpu
