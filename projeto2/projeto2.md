# Solução PAIRS

Job 1 é responsável pelo wordcount, ou seja, contar as palavras dentro do documento e ao final produz um documento com o formato:

[termo] [contador]

O mapper do Job 2 usa a mesma entrada do Job 1, e emite um PairsOfString para cada par de termos coocorrentes. O Reducer do Job 2 usa o método setup para criar um in memory map dos termos e a sua contagem total usando o conjunto de dados produzidos pelo Job 1. Agora que o Reduce tem todos os dados, ele é responsável por calcular o PMI. o Formato de saída é

(termo1, termo2) [PMI]


# Questão 1. Qual é o tempo de execução da implementação “pairs”? Qual é o tempo de execução da implementação “stripes”? Apresente o tempo médio de 10 execuções de cada uma.

117.165 s

*Não consegui implementar o stripes*

# Questão 2. Apresente novamente os resultados da Questão 1, mas desta vez com os Combiners desativados.

118.263 s

# Questão 3. Quantos pares distintos de PMI foram extraídos?

38599

# Questão 4. Qual é o par (x,y) com o PMI mais alto? Justifique intuitivamente este resultado.

(anjou, maine)	10.340645107376782 - Lendo na internet, vi que é um raça de gado


# Questão 5. Quais os três termos que têm o PMI mais alto com “life” e “love”? Quais são os valores do PMI?

(life, save)	4.8705541968380555
(life, man's)	4.204305852259719
(life, long)	3.567572760979865

(love, lysander)	3.866267200296234
(love, rom)	3.454648226613691
(love, valentine)	3.333462669811468


**Questões 6 a 10 utilizaram o documento simplewiki-20161101-pages-articles-multistream-index**

# Questão 6. Qual é o tempo de execução da implementação “pairs”? Qual é o tempo de execução da implementação “stripes”? Apresente o tempo médio de 10 execuções de cada uma.

107.202 s

*Não consegui implementar o stripes*

# Questão 7. Apresente novamente os resultados da Questão 1, mas desta vez com os Combiners desativados.

109.696 s

# Questão 8. Quantos pares distintos de PMI foram extraídos?

4059

# Questão 9. Qual é o par (x,y) com o PMI mais alto? Justifique intuitivamente este resultado.

(elías, piña)	10.652171079110634 - Provavelmente um nome composto

# Questão 10. Quais os três termos que têm o PMI mais alto com “life” e “love”? Quais são os valores do PMI?

(love, you)	5.0661716401108166
(love, with)	3.9825471424081984
(love, song)	3.9623139291496448

(life, the)	2.6803175955086966
(life, of)	2.485188393694909

*Na base utilizada, somente existia duas coocorrencias de life*
