# Contrato de Features — Treino ↔ Avaliação ↔ Serving (FR-007)

**Fonte única no código**: `src/data/feature_contract.py`
**Versão**: 2 (gravada em `NCFRecommender.contract_version`)

Qualquer código que monte a matriz de entrada do modelo (trainer,
`HistoryFeatureLookup.matrix` na avaliação, `RecommendationService._build_matrix`
no serving) DEVE importar a ordem daqui — nunca redeclarar listas de colunas.

## Ordem das colunas (`FEATURE_COLS`)

| Posição | Coluna | Tipo lógico | Origem |
|---------|--------|-------------|--------|
| 0 | `user_idx` | id (embedding) | factorize do preprocess |
| 1 | `item_idx` | id (embedding) | factorize do preprocess |
| 2 | `hour` | contínua | timestamp do evento/contexto |
| 3 | `day_of_week` | contínua | timestamp do evento/contexto |
| 4 | `frequency` | contínua causal | nº de eventos anteriores do usuário |
| 5 | `engagement_score` | contínua causal | soma de pesos anteriores do usuário |
| 6 | `recency_days` | contínua causal | dias desde o evento anterior (−1 = sem histórico) |
| 7 | `view_count` | contínua causal | views anteriores do item |

## Transformação

- Quem monta a matriz entrega **valores crus** (float32) nesta ordem.
- O **modelo** é responsável por: rotear ids desconhecidos para o índice
  unknown (último da tabela de embedding, D4), mapear `item_idx → cat_idx`
  pela tabela interna, e aplicar o `StandardScaler` (ajustado no fit,
  serializado no pickle — D5) às colunas contínuas.
- Consequência: é impossível treino e serving divergirem de escala, porque
  a transformação vive num único objeto serializado.

## Semântica do estado servido

No serving/avaliação as features causais do usuário/item são o estado
"até agora" do histórico disponível:

- `frequency`/`engagement_score`: agregados completos do histórico.
- `recency_days`: dias entre o fim do histórico (referência) e o último
  evento do usuário — no treino, dias desde o evento anterior.
- `view_count`: total de views do item no histórico.
- `hour`/`day_of_week`: contexto da requisição (now) ou do primeiro evento
  de teste do usuário (avaliação).

## Teste de contrato

`tests/test_contract.py` valida que serving e avaliação produzem o mesmo
vetor (colunas não-contextuais) para o mesmo (user, item) sobre o mesmo
histórico, e que o layout é `ID_COLS + CONT_COLS`.

## Evolução

Mudou a ordem/conjunto de colunas? Incremente `CONTRACT_VERSION`,
atualize esta página e o teste de contrato. Modelos serializados carregam
a versão com que foram treinados.
