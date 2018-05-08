#!/usr/bin/bash

echo "Algorithm,Execution_Number,Elapsed_Time_ns,Access_Avg,Access_StdDev,PerThread_Access_Count" > ep4-data.csv;

for i in $(seq 0 13); do 
	for j in $(seq 16 32); do
		./main $((2**$i)) $((2**$j))  >> ep4-data.csv;
		
	done; 
done; 
