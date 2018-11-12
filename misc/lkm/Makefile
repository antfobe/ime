obj-m += test.o

lz4t:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

test:
	sudo insmod test.ko 
	sudo rmmod test

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
