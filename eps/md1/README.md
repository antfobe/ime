INSTITUTO DE MATEMÁTICA E ESTATÍSTICA UNIVERSIDADE DE SÃO PAULO, 2018/03/13
Computação Paralela e Distribuída - MAC5742/MAC0219 2018-1 Prof. Alfredo Goldman 

Desafio de Programação: Predição de Branches

	- O programa feito, speculation_performance.c, funciona iterando um numero 
	randomico em uma recursão em que a ultima iteração sempre é falsa. O codigo
	faz uso de macros que usam predição '__builtin_expect(!!(x), 1)'.
	A função otimizada tenta admite que sempre haverá uma nove recursão, a função 
	incorretamente otimizada admite que a recursão irá encerrar. A função sem
	predição foi deixada como referência, bem como os exemplos de recuperar a
	variavel HOME da memória (sem e com predição).
	- Foi adicionada uma chamada a 'clock()' para evidenciar os efeitos da 
	predição. 
	- Como era de se esperar a predição mais 'correta' tem melhor performance
	que a que só é correta para o final da recursão.
	- Build & run: apenas rode o comando 'make' do diretório do exercício programa.

Referencias:
	

Aluno 	JEAN LUC ANTOINE OLIVIER FOBE
	NUSP 7630573

