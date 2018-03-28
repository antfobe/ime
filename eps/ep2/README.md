INSTITUTO DE MATEMÁTICA E ESTATÍSTICA UNIVERSIDADE DE SÃO PAULO, 2018/03/12
Computação Paralela e Distribuída - MAC5742/MAC0219 2018-1 Prof. Alfredo Goldman 

Mini Exercício Programa: Verificação usando Pthreads. N rãs são colocadas em N posições sucessivas à esquerda de uma série de pedras; M sapos ocupam M quadrados à direita dessa série de pedras. No geral, exitem M + N + 1 pedras, de modo que apenas uma pedra permanece desocupada, conforme ilustrado na figura. No final todos os sapos deverão estar à esquerda e as rãs deverão estar à direita. (Veja o link https://primefactorisation.com/frogpuzzle)

	- Os programa feito, epm2.c funciona inicializando um número M + N de threads
	 e um array de tamanho M + N + 1, de forma a satisfazer a condição inicial do
	 problema 'frogpuzzle'. Foi usado um número de iterações máxima para a 
	validação da situação do array pelas threads de 9001, na variável global
	dead_count. Se houver alguma mudança no array, dead_count é resetado a 0.
	- Atingido o número maximo de iterações as threads são paradas e programa 
	conclui, informando o estado final das posições dos sapos no array e o número
	 de iterações executadas.
	- O número de pedras para o programa pode ser especificado na compilação na
	variável POND_SIZE. Exemplo, para 5 pedras:
	gcc -o epm2 epm2.c -O0 -Wall -std=c99 -lpthread -lm -DPOND_SIZE=5
	- Build & run: apenas rode o comando 'make' do diretório do exercício programa
	 e o programa será compilado e executado para 7 pedras.
	- Observação: há uma condição de corrida que de vez em quando aparece na
	execução que não fui capaz de compreender, fazendo com que dois ou mais sapos
	 ocupem a mesma posição.

Referencias:

	- https://stackoverflow.com/questions/10879420/using-of-shared-variable-by-10-pthreads

Alunos 	JEAN FOBE, EUGENIO GIMENES
