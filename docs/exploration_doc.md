# Exploração e Experimentação de Modelos de Recomendação

## 1. Objetivo da Experimentação

O objetivo desta etapa foi construir e comparar diferentes estratégias de recomendação de produtos para uma base de e-commerce baseada em eventos de navegação. A base contém interações entre usuários e produtos, representadas pelos eventos:

* `view`: visualização do produto;
* `addtocart`: adição do produto ao carrinho;
* `transaction`: compra efetivada.

A experimentação foi conduzida em níveis progressivos de complexidade, partindo de modelos simples baseados em popularidade até modelos neurais baseados em embeddings.

Os algoritmos avaliados foram:

1. Popularidade Ponderada;
2. Basket Analysis com Apriori;
3. Item-Based Collaborative Filtering;
4. Matrix Factorization com ALS;
5. Neural Collaborative Filtering;
6. Two-Tower Model;

---

# 2. Preparação Geral dos Dados

## 2.1 Conversão temporal

A coluna `timestamp` foi convertida para o formato de data e hora, pois a avaliação dos modelos foi realizada de forma temporal.

A lógica aplicada foi:

1. Converter o timestamp Unix em milissegundos para `datetime`;
2. Ordenar os eventos cronologicamente;
3. Separar os dados em treino e teste com base no tempo.

Essa abordagem simula um cenário real de recomendação, no qual o modelo é treinado com interações passadas e avaliado em interações futuras.

---

## 2.2 Separação treino e teste

Foi utilizada uma separação temporal:

```text
passado -> treino
futuro  -> teste
```

Essa escolha evita vazamento de informação, pois nenhum evento futuro é usado para treinar os modelos.

A mesma separação temporal foi mantida para todos os algoritmos, permitindo uma comparação justa entre as estratégias.

---

## 2.3 Definição de relevância

Foram consideradas duas possíveis definições de relevância:

### Relevância ampla

```text
view, addtocart, transaction
```

Neste caso, qualquer interação do usuário com o produto é considerada relevante.

### Relevância forte

```text
addtocart, transaction
```

Neste caso, apenas eventos que indicam maior intenção de compra são considerados relevantes.

Para os modelos neurais, foi adotada preferencialmente a lógica:

```text
positivo = addtocart ou transaction
negativo = view-only
```

Ou seja, uma visualização isolada foi tratada como um sinal negativo ou fraco, desde que o mesmo usuário não tenha posteriormente adicionado o item ao carrinho ou comprado o produto no conjunto de treino.

---

# 3. Métricas de Avaliação

Como o problema é de recomendação Top-K, foram utilizadas métricas próprias de ranking.

## 3.1 Hit Rate@K

O `Hit Rate@K` mede se pelo menos um dos itens recomendados apareceu entre os itens relevantes do usuário no teste.

Formalmente:

```text
Hit Rate@K = 1 se houver pelo menos um acerto no Top-K
Hit Rate@K = 0 caso contrário
```

Essa métrica responde à pergunta:

> O modelo conseguiu acertar pelo menos uma recomendação relevante para o usuário?

É uma métrica simples e útil para medir cobertura de acerto por usuário.

---

## 3.2 Precision@K

A `Precision@K` mede a proporção de itens recomendados que são relevantes.

```text
Precision@K = itens relevantes recomendados / K
```

Essa métrica responde:

> Dos K itens recomendados, quantos realmente eram relevantes?

Ela penaliza listas de recomendação com muitos itens irrelevantes.

---

## 3.3 Recall@K

O `Recall@K` mede a proporção de itens relevantes do usuário que foram recuperados pelo modelo.

```text
Recall@K = itens relevantes recomendados / total de itens relevantes do usuário
```

Essa métrica responde:

> Dos itens que o usuário realmente interagiu no futuro, quantos o modelo conseguiu recuperar?

É uma métrica importante quando o objetivo é maximizar a chance de capturar itens relevantes.

---

## 3.4 NDCG@K

O `NDCG@K` avalia não apenas se o modelo acertou, mas também a posição dos acertos na lista recomendada.

Acertos nas primeiras posições recebem maior peso do que acertos nas últimas posições.

Essa métrica responde:

> O modelo está colocando os itens relevantes no topo da lista?

É uma das métricas mais importantes em sistemas de recomendação, pois a ordem dos itens recomendados impacta diretamente a experiência do usuário.

---

# 4. Popularidade Ponderada

## 4.1 O que o algoritmo calcula

O modelo de popularidade calcula um ranking global de produtos com base na frequência e intensidade das interações.

Foi atribuída uma pontuação diferente para cada tipo de evento:

```text
view        = 1
addtocart   = 5
transaction = 10
```

Assim, produtos comprados recebem maior peso do que produtos apenas visualizados.

---

## 4.2 Lógica aplicada passo a passo

1. Selecionar apenas os dados de treino;
2. Mapear cada tipo de evento para um peso numérico;
3. Agrupar os dados por `itemid`;
4. Somar os pesos dos eventos de cada produto;
5. Ordenar os produtos pelo score de popularidade;
6. Selecionar os Top-K produtos mais populares;
7. Recomendar a mesma lista Top-K para todos os usuários;
8. Comparar as recomendações com os itens relevantes do teste;
9. Calcular `Hit Rate@K`, `Precision@K`, `Recall@K` e `NDCG@K`;
10. Armazenar os resultados em uma estrutura tabular para comparação posterior.

---

## 4.3 Para que serve

Este modelo serve como baseline inicial. Ele responde:

> Quais são os produtos mais populares globalmente?

É útil como primeiro ponto de comparação, pois qualquer modelo mais sofisticado deve superar esse baseline para justificar sua complexidade.

---

## 4.4 Vantagens

* Simples de implementar;
* Rápido de treinar;
* Fácil de explicar;
* Funciona bem para produtos muito populares;
* Útil como fallback para usuários sem histórico.

---

## 4.5 Desvantagens

* Não é personalizado;
* Recomenda os mesmos produtos para todos os usuários;
* Favorece produtos já populares;
* Não captura preferências individuais;
* Pode reforçar viés de popularidade.

---

# 5. Basket Analysis com Apriori

## 5.1 O que o algoritmo calcula

O Apriori identifica conjuntos de produtos que aparecem frequentemente juntos nas interações dos usuários.

A partir desses conjuntos, são geradas regras de associação do tipo:

```text
produto A -> produto B
```

A interpretação é:

> Usuários que interagiram com o produto A também tendem a interagir com o produto B.

---

## 5.2 Lógica aplicada passo a passo

1. Selecionar os dados de treino;
2. Definir quais eventos serão usados para formar as cestas;
3. Agrupar os produtos por usuário, formando uma cesta por `visitorid`;
4. Remover produtos duplicados dentro da mesma cesta;
5. Filtrar cestas com menos de dois produtos;
6. Converter as cestas para uma matriz transacional one-hot;
7. Aplicar o algoritmo Apriori para encontrar itemsets frequentes;
8. Gerar regras de associação a partir dos itemsets;
9. Filtrar regras por `confidence` e `lift`;
10. Criar um dicionário de recomendação baseado nas regras;
11. Para cada usuário, buscar regras cujo antecedente esteja em seu histórico;
12. Recomendar os consequentes das regras;
13. Remover itens já vistos pelo usuário;
14. Completar recomendações com fallback de popularidade, se necessário;
15. Avaliar as recomendações contra o conjunto de teste;
16. Armazenar as métricas.

---

## 5.3 Para que serve

O Apriori serve para capturar relações explícitas de coocorrência entre produtos.

Ele é útil quando queremos responder:

> Quais produtos costumam aparecer juntos no comportamento dos usuários?

---

## 5.4 Vantagens

* Fácil de interpretar;
* Gera regras explicáveis;
* Não exige embeddings nem redes neurais;
* Útil para análise de cesta;
* Pode revelar padrões de associação entre produtos.

---

## 5.5 Desvantagens

* Sofre com alta esparsidade;
* Pode gerar poucas regras relevantes em bases muito grandes;
* Pode gerar muitas regras se o suporte mínimo for muito baixo;
* Não lida bem com usuários com pouco histórico;
* Regras baseadas em `view` podem ser ruidosas.

---

# 6. Item-Based Collaborative Filtering

## 6.1 O que o algoritmo calcula

O Item-Based Collaborative Filtering calcula similaridade entre produtos com base nos usuários que interagiram com eles.

A lógica é:

```text
se muitos usuários interagiram com A e B,
então A e B são considerados similares.
```

Depois, para cada usuário, o modelo recomenda itens parecidos com os produtos que ele já consumiu ou interagiu.

---

## 6.2 Lógica aplicada passo a passo

1. Selecionar os dados de treino;
2. Atribuir pesos aos eventos;
3. Agregar as interações por par `visitorid` e `itemid`;
4. Criar índices numéricos para usuários e produtos;
5. Construir uma matriz esparsa usuário-item;
6. Transpor a matriz para obter uma representação item-usuário;
7. Calcular similaridade entre itens usando similaridade do cosseno;
8. Para cada usuário, identificar os itens já interagidos;
9. Somar os scores dos itens similares;
10. Remover itens já vistos pelo usuário;
11. Retornar os Top-K itens com maior score;
12. Aplicar fallback de popularidade para usuários sem histórico;
13. Avaliar as recomendações no teste;
14. Armazenar as métricas.

---

## 6.3 Para que serve

Esse algoritmo serve para gerar recomendações personalizadas com base na similaridade entre produtos.

Ele responde:

> Quais produtos são parecidos com aqueles que o usuário já demonstrou interesse?

---

## 6.4 Vantagens

* Mais personalizado que popularidade;
* Simples em comparação com modelos latentes;
* Não exige treinamento iterativo complexo;
* Pode funcionar bem quando há coocorrência suficiente entre itens;
* Boa interpretabilidade: recomendações podem ser explicadas por itens similares.

---

## 6.5 Desvantagens

* Sofre com matriz usuário-item muito esparsa;
* Itens com poucas interações têm similaridade pouco confiável;
* Usuários novos precisam de fallback;
* Pode ser custoso calcular similaridade item-item em catálogos grandes;
* Não captura fatores latentes complexos.

---

# 7. Matrix Factorization com ALS

## 7.1 O que o algoritmo calcula

O ALS fatoriza a matriz usuário-item em duas matrizes latentes:

```text
usuários -> embeddings latentes
itens    -> embeddings latentes
```

A interação entre um usuário e um item é estimada pelo produto entre seus vetores.

Em dados implícitos, os valores da matriz representam confiança na interação, não uma nota explícita.

---

## 7.2 Lógica aplicada passo a passo

1. Selecionar os dados de treino;
2. Atribuir pesos aos eventos;
3. Agregar interações por par `visitorid` e `itemid`;
4. Criar índices numéricos para usuários e itens;
5. Construir uma matriz esparsa usuário-item;
6. Multiplicar a matriz por um fator `alpha` para representar confiança;
7. Transpor a matriz para o formato esperado pela biblioteca `implicit`;
8. Treinar o modelo ALS;
9. Para cada usuário, gerar scores para os itens candidatos;
10. Filtrar itens já vistos pelo usuário;
11. Retornar os Top-K itens recomendados;
12. Usar popularidade como fallback em casos de cold start;
13. Avaliar as recomendações com as métricas Top-K;
14. Armazenar os resultados.

---

## 7.3 Para que serve

O ALS serve para capturar padrões latentes de preferência.

Ele responde:

> Quais itens são compatíveis com o perfil latente do usuário?

---

## 7.4 Vantagens

* Mais robusto que similaridade simples;
* Captura padrões latentes;
* Funciona bem com feedback implícito;
* Escala melhor que muitos métodos colaborativos tradicionais;
* Pode superar baselines simples em bases esparsas.

---

## 7.5 Desvantagens

* Ainda sofre com cold start;
* Exige ajuste de hiperparâmetros;
* Menos interpretável que Apriori ou Item-Based CF;
* Usuários com poucas interações geram embeddings fracos;
* Itens novos sem interação não são bem tratados.

---

# 8. Neural Collaborative Filtering

## 8.1 O que o algoritmo calcula

O Neural Collaborative Filtering aprende embeddings de usuários e itens usando uma rede neural.

A arquitetura utilizada foi simples:

```text
user_id -> user embedding
item_id -> item embedding
concatenação -> MLP -> probabilidade de interação
```

O modelo foi treinado como uma tarefa binária:

```text
1 = interação positiva
0 = interação negativa
```

---

## 8.2 Lógica aplicada passo a passo

1. Selecionar os dados de treino;
2. Definir eventos positivos e negativos;
3. Considerar `addtocart` e `transaction` como positivos;
4. Considerar `view-only` como negativo;
5. Remover dos negativos pares usuário-item que também aparecem como positivos;
6. Amostrar negativos para controlar o desbalanceamento;
7. Concatenar positivos e negativos em um dataset único;
8. Criar índices numéricos para usuários e itens;
9. Dividir o dataset em treino e validação interna;
10. Construir um Dataset PyTorch;
11. Criar DataLoaders;
12. Definir embeddings de usuários e itens;
13. Concatenar embeddings;
14. Passar a concatenação por uma MLP simples;
15. Treinar com `BCEWithLogitsLoss`;
16. Monitorar perda de validação;
17. Aplicar early stopping;
18. Gerar scores para todos os itens candidatos;
19. Remover itens já vistos pelo usuário;
20. Retornar recomendações Top-K;
21. Avaliar no conjunto de teste;
22. Armazenar os resultados.

---

## 8.3 Para que serve

O NCF serve para aprender relações não lineares entre usuários e itens.

Ele responde:

> Qual a probabilidade de um usuário interagir fortemente com determinado item?

---

## 8.4 Vantagens

* Permite modelar relações não lineares;
* Aprende embeddings diretamente dos dados;
* Pode superar fatoração linear;
* Flexível para incluir novas features;
* Boa ponte entre recomendação clássica e deep learning.

---

## 8.5 Desvantagens

* Mais complexo de treinar;
* Exige negative sampling;
* Sensível à definição de positivo e negativo;
* Pode overfitar em bases esparsas;
* Exige mais tempo computacional;
* Menos interpretável.

---

# 9. Two-Tower Model

## 9.1 O que o algoritmo calcula

O Two-Tower Model aprende duas representações separadas:

```text
torre do usuário -> embedding do usuário
torre do item    -> embedding do item
```

A compatibilidade entre usuário e item é calculada por produto escalar ou similaridade entre embeddings.

---

## 9.2 Lógica aplicada passo a passo

1. Selecionar os dados de treino;
2. Definir interações positivas como `addtocart` e `transaction`;
3. Definir interações negativas como `view-only`;
4. Remover pares usuário-item que aparecem como positivos da base negativa;
5. Amostrar negativos;
6. Criar índices para usuários e itens;
7. Construir Dataset e DataLoader no PyTorch;
8. Criar uma torre neural para usuários;
9. Criar uma torre neural para itens;
10. Gerar embeddings separados para usuário e item;
11. Normalizar os embeddings;
12. Calcular o score pelo produto escalar;
13. Treinar com `BCEWithLogitsLoss`;
14. Aplicar early stopping com base na perda de validação;
15. Pré-computar embeddings dos itens;
16. Para cada usuário, calcular similaridade com todos os itens;
17. Remover itens já vistos;
18. Retornar os Top-K itens;
19. Avaliar no teste;
20. Armazenar as métricas.

---

## 9.3 Para que serve

O Two-Tower serve principalmente para sistemas de recuperação de candidatos em larga escala.

Ele responde:

> Quais itens estão mais próximos do usuário em um espaço vetorial compartilhado?

---

## 9.4 Vantagens

* Arquitetura próxima de sistemas reais de recomendação;
* Permite pré-computar embeddings de itens;
* Facilita busca vetorial;
* Escala melhor para recomendação em catálogos grandes;
* Pode incorporar features adicionais nas torres.

---

## 9.5 Desvantagens

* Mais complexo que NCF simples;
* Exige cuidado com negative sampling;
* Pode performar mal se houver pouco histórico por usuário;
* Requer mais engenharia;
* Ainda sofre com cold start se usar apenas IDs.

---


# 10. Considerações Finais

A experimentação foi organizada em ordem crescente de complexidade.

A Popularidade Ponderada serviu como baseline global. O Apriori introduziu regras de associação entre produtos. O Item-Based CF trouxe personalização baseada em similaridade entre itens. O ALS adicionou fatores latentes para capturar padrões ocultos. O NCF introduziu modelagem neural não linear. O Two-Tower aproximou o projeto de uma arquitetura moderna de recuperação de candidatos. Por fim, a inclusão da Category Tree permitiu enriquecer a representação dos itens com informação hierárquica do catálogo.

Essa progressão permite avaliar não apenas qual modelo apresenta melhor desempenho, mas também qual nível de complexidade é realmente justificável diante das características da base, como alta esparsidade, muitos usuários com poucas interações e forte predominância de eventos de visualização.

Algo a ser testado e avaliado posteriormente após fluxo DVC montado e validado, é utilizar um filtro de interações minimas para validar o cliente para reduzir ruídos adicionados a clientes tem uma única interação e apenas relacionada a `view`.
