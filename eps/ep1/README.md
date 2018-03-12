INSTITUTO DE MATEMÁTICA E ESTATÍSTICA UNIVERSIDADE DE SÃO PAULO, 2018/03/12
Computação Paralela e Distribuída - MAC5742/MAC0219 2018-1 Prof. Alfredo Goldman 

Mini Exercício Programa: Utilização das Memórias Caches Descrição Os Processadores atuais têm diferentes níveis de cachês. Tais níveis ajudam a incrementar a largura de banda da comunicação com a memória principal do computador (RAM), reduzindo os efeitos do bottleneck de Von Neumann. Neste mini Exercício Programa, cada estudante deverá mostrar o aproveitamento dos níveis de cache em um programa computacional. Esse programa pode ser codificado em qualquer linguagem de programação, mas recomenda-se fortemente qualquer versão do C, C++, ou Fortran.

	- Os programas feitos, epm1a.c e epm1b.c, funcionam percorrendo um array char 
	gigante M de tamanho 1024³. O primeiro programa, mais lento, percorre o array 
	em passos variando de tamanho 'k' 1024² a 1024 (loop 1), o segundo, epm1b, de 
	1024 a 1 (loop 2). Foi adicionado um limite MAX_STEP para impedir o programa 
	'a', epm1a.c, de demorarem mais tempo por simplesmente percorrer mais casas 
	do array.
	- Foi considerado que a melhor forma para chamar o cache era atualizar o 
	valor da casa do array (M[i]+=3;). 
	- Como era de se esperar o programa 'b' é mais rápido que o 'a'.
	- Build & run: apenas rode o comando 'make' do diretório do exercício programa.

Referencias:
	

Aluno 	JEAN LUC ANTOINE OLIVIER FOBE
	NUSP 7630573

