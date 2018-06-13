#!/bin/bash

MNUM=$1;
FNAME=$2;
re='^[0-9]+$'

if [[ !$1 =~ $re ]] ; then
	MNUM=2;
fi

if [[ -z "$FNAME" ]] ; then
	FNAME="sample.txt"; 
fi

printf "$MNUM\n" > $FNAME;
for i in $(seq 1 $MNUM) ; do
	printf "∗∗∗\n" >> $FNAME;
	for j in $(seq 1 3) ; do
		printf "$(( (RANDOM % 10) + (RANDOM % 15) )) $(( (RANDOM % 10) + (RANDOM % 15) )) $(( (RANDOM % 10) + (RANDOM % 15) ))\n" >> $FNAME;
	done
done

printf "∗∗∗\n" >> $FNAME;
