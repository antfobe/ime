#!/bin/bash

RED='\033[0;31m';
GRN='\033[1;32m';
YEL='\033[1;33m';
BLU='\033[1;34m';
NC='\033[0m';

## Initialize .csv header
echo "SIZE_VECTOR, NUM_THREADS, NUM_IFS, AVG5_TIME(s)" >> contention_out.csv;

## Outer loop, iterate over array sizes up to 2^28;
for j in $(seq 0 28) ; do

	## Inner loop, iterate over number of threads up to 4096 (limit per process is about ~>7000);
	for k in $(seq 0 12) ; do
		## Check progress -- in colors !! --
		printf "\t\t${YEL}Execution: ${GRN}$i ${BLU}$j ${NC}\n";
		## Run iteration, append .csv
		./contention.sh $((2**$j)) $((2**$k)) >> contention_out.csv;
	done;
done;
