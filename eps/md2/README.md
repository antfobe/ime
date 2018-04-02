INSTITUTO DE MATEMÁTICA E ESTATÍSTICA UNIVERSIDADE DE SÃO PAULO, 2018/04/02
Computação Paralela e Distribuída - MAC5742/MAC0219 2018-1 Prof. Alfredo Goldman 

Desafio de Programação: Hyperthreading 

	- O conjunto de programas pesquisado e ligeiramente modificado disponível
	em https://github.com/tsuna/contextswitch, faz um benchmark de 
	performance baseado em chaveamentos de contexto de processos e threads em 
	processadores. As razões por se ter optado por usar esse código para
	responder ao problema proposto pelo desafio são as mudanças de contexto 
	de threads feitas no caso com affinidade de cpu em um tempo intermediário,
	 que tiveram performance ligeramente maior do que nos casos sem afinidade 
	de CPU.
	- Recapitulando o que está acontecendo, threads e processos estão sendo
	mudados de contexto, uma atividade que pode vir a atrapalhar a performance
	 dos programas sendo executados, uma vez que o chaveamento adiciona 
	overhead no processamento. No caso de hyperthreading, essa mudança deve 
	ser menor para para mudanças entre núcleos lógicos (hardware threads)
	e núcleos físicos na CPU - isso pode ser observado na execução do
	benchmark para mudanças de contexto sem afinidade na CPU, ou com afinidade
	a um núcleo físico (CPU 0).
	- Setando afinidade de CPU, estamos forçando todas as mudanças de contexto
	serem feitas no mesmo núcleo, independente se lógico ou físico.
	- Podemos concluir que para chaveamentos com frequencia moderada, o
	recurso de hyperthreading será alocado para outra atividade mais intensiva
	 ocasionando uma leve perda de performance na execução de um programa
	semelhante ao de benchmark.
 
Referencias:

	- https://software.intel.com/en-us/articles/methods-to-utilize-intels-hyper-threading-technology-with-linux
	- https://github.com/tsuna/contextswitch

Output da execução no meu pc:

	model name : Intel(R) Core(TM) i3 CPU M 350 @ 2.27GHz
	1 physical CPUs, 2 cores/CPU, 2 hardware threads/core = 4 hw threads total
			-- No CPU affinity --
	16777216 system calls in 6434586615ns (383.5ns/syscall)
	2097152 process context switches in 6378058987ns (3041.3ns/ctxsw)
	2097152  thread context switches in 6423851979ns (3063.1ns/ctxsw)
	2097152  thread context switches in 413438440ns (197.1ns/ctxsw)
			-- With CPU affinity --
	16777216 system calls in 6547561238ns (390.3ns/syscall)
	2097152 process context switches in 7065319249ns (3369.0ns/ctxsw)
	2097152  thread context switches in 6098017466ns (2907.8ns/ctxsw)
	2097152  thread context switches in 1166615532ns (556.3ns/ctxsw)
			-- With CPU affinity to CPU 0 --
	16777216 system calls in 6558742612ns (390.9ns/syscall)
	2097152 process context switches in 6284015199ns (2996.5ns/ctxsw)
	2097152  thread context switches in 6485067655ns (3092.3ns/ctxsw)
	2097152  thread context switches in 419974994ns (200.3ns/ctxsw)

Alunos 	JEAN FOBE, EUGENIO GIMENES
