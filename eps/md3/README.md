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
	
 
Alguns resultados:
	
	#ifs	|	array size (bits)	|	#threads	|	avg5(time) (s)	
	--------|-------------------------------|-----------------------|-------------------------
	0	|			2	|		2	|		
	0	|			
	0	|
	5	|
	5	|
	5	|
	5	|
	9	|
	9	|
	9	|

Alunos 	JEAN FOBE

