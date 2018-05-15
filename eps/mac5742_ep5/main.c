#include "ep5.h"

int main(){

	A = readM("A.txt");
	B = readM("B.txt");
	omp_mmul(m, n, p, A, B, C);
	if(!writeC("C.txt", C)) {DBG("failed to write C matrix to file"); exit(-1);}

	return 0;
}
