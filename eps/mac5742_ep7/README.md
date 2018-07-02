INSTITUTO DE MATEMÁTICA E ESTATÍSTICA UNIVERSIDADE DE SÃO PAULO, 2018/07/02
Computação Paralela e Distribuída - MAC5742/MAC0219 2018-1 Prof. Alfredo Goldman

Implementação:
	- O programa feito, no arquivo ep7.cu, funciona conforme o enunciado do 
	exercício, isto é, o programa compilado (make ep7) ep7, recebe como 
	parametro < N, M, K > e imprime em stdout o resultado da integração.
	- Notei que quando M e k se aproximam o resultado difere mais do
	esperado (~1).
	Também não esperava ter que implementar	a função em um kernel para ter 
	ainda que reduzir o resultado...
	- Build & run: apenas rode o comando 'make' do diretório do exercício 
	programa e o programa será compilado. Rode 'make test' para executar
	o programa com argumentos (N,M,k) = (65536,8192,1024).

Alunos 	JEAN FOBE
	EUGENIO GIMENES 
