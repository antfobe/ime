INSTITUTO DE MATEMÁTICA E ESTATÍSTICA UNIVERSIDADE DE SÃO PAULO, 2018/05/18
Computação Paralela e Distribuída - MAC5742/MAC0219 2018-1 Prof. Alfredo Goldman 

Mini Exercício Programa: Multiplicação paralela de matrizes. Sejam A ∈ R m x p , B ∈ R p xn matrizes de entradas reais com m linhas e p colunas, e p linhas e n colunas, respectivamente. Neste exercício programa deve ser implementado um ou mais algoritmos para calcular: 
		C = AB
da maneira mais eficiente possível em precisão dupla, ou seja, número de ponto flutuante de 64-bits. Em C/C++, este tipo corresponde a um double. Em Fortran, este tipo corresponde a um DOUBLE PRECISION.

	- O programa feito, em arquivos ep5.c, ep5.h e ep5_main.c, funciona conforme
	o enunciado do exercício e com a opção de não ler arquivos de entrada.
	O programa compilado (make ep5) ep5, recebe como parametros < p|o > < A.txt >
	< B.txt > < C.txt > e imprime no terminal o tempo tomado com a multiplicação
	de matrizes em si (desprezando IO). O programa também pode receber como
	entrada < P|O > < m > < n > < p >, onde m,n,p são dimensões as matrizes - 
	com esses parametros o programa faz as operações de multiplicação com dados
	'sujos' da memória.
	- O código feito também permite a compilação com variável de debug (make 
	debug), mas cuidado, porque dependendo do tamanho da matriz os outputs vão
	ser muito verbosos.  
	- Outro detalhe é que para a execução utilizando pthreads, vai ser criado um
	número de threads igual ao número de núcleos da máquina, o que pode vir a
	ocasionar falta de resposta do computador na execução do programa.
	- Build & run: apenas rode o comando 'make' do diretório do exercício programa
	 e o programa será compilado e executado com dados 'sujos' da memória nas
	opções 'P' e 'O' e com (m,n,p) = (1024,512,2048).

Alunos 	JEAN FOBE
