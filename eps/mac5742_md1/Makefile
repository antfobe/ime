LIBS =
CC = gcc
CFLAGS = -O3 -Wall -lbsd -std=c99
OPTCFLAGS =-O0 -Wall -lbsd -std=c99

default: run

run: spec
	./spec_perf

runopt: opt
	./spec_opt

all: spec

spec: speculation_performance.c
	$(CC) $(CFLAGS) $(LIBS) -o spec_perf speculation_performance.c
opt: speculation_performance.c
	$(CC) $(OPTCFLAGS) $(LIBS) -o spec_opt speculation_performance.c

license:
	-less LICENSE

clean:
	-rm -f spec_perf
	-rm -f spec_opt
