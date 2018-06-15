INSTITUTO DE MATEMÁTICA E ESTATÍSTICA UNIVERSIDADE DE SÃO PAULO, 2018/06/15
Computação Paralela e Distribuída - MAC5742/MAC0219 2018-1 Prof. Alfredo Goldman

Exercício Programa: A redução é uma estratégia de divisão e conquista, e consiste em definir um operador sobre todos os elementos de uma coleção de forma a particionar o problema, resolver as partes, e combiná-las para obter a solução final. Neste exercício programa, você deverá implementar uma operação específica de redução em CUDA:
	- Seja Z 3×3 o conjunto das matrizes 3×3 com entradas inteiras. Sejam A, B ∈ Z 3×3 e denote aij como o elemento correspondente a i-ésima linha e j-ésima coluna de A, e analogamente bij para B. Definiremos o operador ⊕ sobre Z 3×3 da seguinte forma: A ⊕ B = min(aij,bij);

Implementação:
	- O programa feito, no arquivo ep6.cu, funciona conforme o enunciado do 
	exercício, isto é, o programa compilado (make ep6) main, recebe como 
	parametro < filepath > (O arquivo que contem as matrizes) e imprime em
	stdout o resultado da computação (matriz 3x3 de mínimos).
	- Também foi feito um script que gera as matrizes (make sample || make 
	test) gensample.sh. 
	- Um detalhe observado, no entanto, é que o programa funciona somente 
	para um número de matrizes menor que 115. Para um número maior ou
	igual a 115 ocorre o seguinte erro: 'invalid configuration argument'.
	- Build & run: apenas rode o comando 'make' do diretório do exercício 
	programa e o programa será compilado. Rode 'make test' para gerar um 
	arquivo sample.txt com número de matrizes menor que 115 que será usado 
	na redução.

Alunos 	JEAN FOBE
	EUGENIO GIMENES 
