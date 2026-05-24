# Model Card — Sistema de Recomendação de Produtos

> Preencher após escolher o dataset e completar o treinamento.

## Detalhes do Modelo

| Campo | Valor |
|-------|-------|
| **Nome** | recommender |
| **Versão** | A definir |
| **Tipo** | MLP (PyTorch) |
| **Tarefa** | Recomendação binária (feedback implícito) |
| **Framework** | PyTorch |
| **Estágio no MLflow** | A definir |

## Uso Pretendido

- **Uso principal**: Recomendar produtos a usuários de e-commerce com base no histórico de navegação/compra.
- **Fora do escopo**: Usuários ou itens sem histórico de interações (cold-start).

## Dataset

| Campo | Valor |
|-------|-------|
| **Nome** | A definir |
| **Fonte** | A definir |
| **Tamanho** | A definir |
| **Splits** | treino / validação / teste |

## Desempenho

> A preencher após a avaliação.

| Métrica | Treino | Validação | Teste |
|---------|--------|-----------|-------|
| ROC-AUC | — | — | — |
| Average Precision | — | — | — |
| F1 | — | — | — |
| Precisão | — | — | — |
| Recall | — | — | — |

## Comparação com Baselines

| Modelo | ROC-AUC | F1 |
|--------|---------|----|
| Popularidade | — | — |
| Regressão Logística | — | — |
| **MLP (nosso)** | — | — |

## Limitações

- O desempenho cai para usuários/itens com menos interações do que `min_interactions`.
- Não modela padrões sequenciais (planejado: modelo baseado em embeddings).
- Treinado com feedback implícito — as amostras negativas são sintéticas.

## Vieses

- **Viés de popularidade**: itens com mais interações estão super-representados no treino.
- **Viés de recência**: se a ordenação temporal for usada, interações mais antigas podem ser subponderadas.

## Detalhes do Treinamento

- Otimizador: Adam
- Função de perda: BCEWithLogitsLoss
- Early stopping: paciência de 5 épocas
- Taxa de amostragem negativa: 4:1

## Considerações Éticas

- Nenhuma informação pessoal identificável (PII) é usada nas features do modelo.
- A diversidade das recomendações deve ser monitorada para evitar bolhas de filtro.
