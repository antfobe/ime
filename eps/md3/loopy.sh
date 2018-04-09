#!/bin/sh

RED='\033[0;31m';
GRN='\033[1;32m';
YEL='\033[1;33m';
BLU='\033[1;34m';
NC='\033[0m';

for i in {1,16,24,28} ; do
	for j in {1,5,6,7,8,9,10} ; do
	 	printf "\t\t${YEL}Execution: ${GRN}$i ${BLU}$j ${NC}\n";
		echo -e "\t\tExecution: Array size:[$((2 ** $i))] Threads:[$((2 ** $j))]\n" >> contention.out;
		./contention.sh $((2 ** i)) $((2 ** j)) >> contention.out;
	done;
done;
