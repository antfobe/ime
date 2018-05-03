#!/usr/bin/bash

echo "Algorithm,Execution_Number,Elapsed_Time_ns,Access_Avg,Access_StdDev,PerThread_Access_Count" > ep4-data.csv;

for i in $(seq 1 8192); do 
	for j in $(seq 1 64); do
		./main $i $((2**$j))  >> ep4-data.csv;
		
	done; 
done; 
