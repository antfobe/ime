INSTITUTO DE MATEMÁTICA E ESTATÍSTICA UNIVERSIDADE DE SÃO PAULO, 2018/04/09
Computação Paralela e Distribuída - MAC5742/MAC0219 2018-1 Prof. Alfredo Goldman 

Desafio de Programação: Contenção

	- Pelo script fornecido, 'contentions.sh' entende-se que será compilado 
	e executado um programa em C (Standard C99), usando OpenMP para 
	paralelizar operações feitas em um vetor. O tamanho do vetor é 
	especificado como argumento do script, bem como o número de threads. 
	Outro parâmetro da execução é estático: a quantidade de condicionais, 
	'if's, que são colocados antes da região crítica do problema. O script 
	varia a quantidade de 'if's entre 0 e 9 para os parâmetros fornecidos 
	na entrada. 
	- Para observar os efeitos da contenção fiz um script adicional 
	'loopy.sh', que rodará o 'contentions.sh' com valores de potências de 2
	para o argumento do vetor de 1,16,24,28 (ou seja 2^1, 2^16, ..., 2^28) 
	e argumento de threads com valores (potências de 2) de 1,5,6,7,8,9,10. 
	Os resultados obtidos estão no arquivo 'contentions.out'.
	- Pude notar que a quantidade de threads adiciona um overhead continuo 
	no processamento para o meu caso, então a perda de performance devido
	a esse fator foi de taxa logarítmica. Já o tamanho do vetor, para 2^28,
	entrou no Swap do PC, piorando exponencialmente a performance. Vetores
	menores contribuiram com perdas proporcionais ao seu tamanho (~2^n).
	Infelizmente devido a arquitetura do processador do meu computador
	(Intel i3), não foi observado o efeito de otimização esperado dos 'if's
	em encadeados, ou seja só houve melhoria de performance entre o caso 
	sem 'if'. Curiosamente, também não houve perda de performance para o 
	acréscimo de dois ou mais 'if's.
	- Pela demora da execução para vetores grandes (>45 min), não foram
	feitos testes com vetores de tamanho maior que 2^28. 
	
 
Alguns resultados:
	
	#ifs	|	array size (sizeof int)	|	#threads	|	avg5(time) (s)	
	--------|-------------------------------|-----------------------|-------------------------
	0	|		2		|	2		|	0.000003	
	0	|		16777216	|	128		|	2.301797
	0	|		268435456	|	1024		|	33.707113
	5	|		2		|	2		|	0.000003
	5	|		16777216	|	128		|	0.034981
	5	|		16777216	|	1024		|	0.041553
	5	|		268435456	|	1024		|	0.566707
	9	|		2		|	2		|	0.000002
	9	|		16777216	|	128		|	0.035693
	9	|		268435456	|	1024		|	0.555212

	- Importante notar que mesmo o enunciado do desafio pedindo relevância 
	estatística, dada a natureza do problema a massa de dados obtida é
	pequena e os dados carregam um 'bias' de terem sido obtidos no mesmo
	computador. Os resultados são coerentes mas não podem ser generalizados.

Alunos 	JEAN FOBE


## Apendice
#!/bin/bash

echo "SIZE_VECTOR, NUM_THREADS, NUM_IFS, AVG5_TIME(s)" >> contention_out.csv

for j in $(seq 0 28) ; do
		for k in $(seq 0 12) ; do
			./contention.sh $((2**$j)) $((2**$k)) >> contention_out.csv
		done
	done


#!/bin/bash

C1="
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define IF if (v[i] > max)

#define NUM_EXEC 8

static void populate_vector(size_t N, int v[])
{
    size_t i;
    srand(1337);

    for (i = 0; i < N; ++i)
        v[i] = rand();
}

static double contention_test(size_t N, int T, const int v[])
{
    size_t i;
    int max = -1;
    double t0, t1;

    omp_lock_t mutex;

    omp_init_lock(&mutex);

    t0 = omp_get_wtime();
    #pragma omp parallel for private(i) num_threads(T)
    for (i = 0; i < N; ++i)
    {
        /*--*/
"

C2="
        /*--*/
        {
            omp_set_lock(&mutex);
            {
                if (v[i] > max)
                    max = v[i];
            }
            omp_unset_lock(&mutex);
        }
    }
    t1 = omp_get_wtime();

    return (t1-t0);
}

static double avg(int n, const double v[])
{
    int i;
    double sum = 0.;
    for (i = 0; i < n; ++i)
        sum += v[i];
    return sum/n;
}

int main(int argc, char* argv[])
{
    static double times[NUM_EXEC];
    int* vector = NULL;

    size_t N;
    int T, i;

    if (argc != 3)
    {
        fprintf(stdout, \"Usage: %s <vector_size> <number_of_threads>\\n \", argv[0]);
        return 1;
    }

    N = atoll(argv[1]);
    T = atoi(argv[2]);
    
    vector = (int*) malloc(N*sizeof(int));

    if (!vector)
    {
        fprintf(stdout, \"Failed to allocate memory. Exiting...\\n\");
        return 2;
    }

    populate_vector(N, vector);

    /*throw away first execution*/
    times[0] = contention_test(N, T, vector);
    for (i = 0; i < NUM_EXEC; ++i)
        times[i] = contention_test(N, T, vector);
    
    //fprintf(stdout, \" Average of %d executions: %lf s\\n\", NUM_EXEC, avg(NUM_EXEC, times));
    fprintf(stdout, \"%lf \\n\", avg(NUM_EXEC, times));
    
    free(vector);
    return 0;    
}
"

###############################################################################

SIZE_VECTOR=$1
NUM_THREADS=$2

generate_ifs() {
    c_ifs=""
#    for ((i=0; i<$1; i++)); do
    for i in $(seq 1 $1); do
        c_ifs+="IF "
    done
}

generate_c() {
    generate_ifs $1
    echo "$C1$c_ifs$C2" > $2
}

run_for_if() {
    #for ((num_ifs=0; num_ifs<9; num_ifs++)); do
    for num_ifs in $(seq 0 9); do
	    echo -n "$1, $2, $((2 **$num_ifs)), "
        #generate_c $num_ifs "temp.c"
	generate_c $((2 ** $num_ifs)) "temp.c"
        gcc -Wall -O0 -std=c99 -fopenmp -o temp temp.c
        ./temp $1 $2
    done
}

if [ -z "$SIZE_VECTOR" ] || [ -z "$NUM_THREADS" ]; then
    echo "Usage: ${0##*/} <vector_size> <num_threads>"
    exit
fi
