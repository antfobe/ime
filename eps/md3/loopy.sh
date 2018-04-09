#!/bin/sh

RED='\033[0;31m';
GRN='\033[1;32m';
YEL='\033[1;33m';
BLU='\033[1;34m';
NC='\033[0m';

## Outer loop, iterate over 2,2^16,2^24,2^28 array sizes;
for i in {1,16,24,28} ; do

	## Inner loop, iterate over, 2,32,64,128,256,512,1024 number of threads;
	for j in {1,5,6,7,8,9,10} ; do
		## Check progress -- in colors !! --
	 	printf "\t\t${YEL}Execution: ${GRN}$i ${BLU}$j ${NC}\n";
		## Put some identifiers in output.
		echo -e "\t\tExecution: Array size:[$((2 ** $i))] Threads:[$((2 ** $j))]\n" >> contention.out;
		## Run iteration
		./contention.sh $((2 ** i)) $((2 ** j)) >> contention.out;
	done;
done;
