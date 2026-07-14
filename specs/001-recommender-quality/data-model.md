# Data Model — Recomendador que aprende preferências

**Feature**: `001-recommender-quality` · **Plan**: [`plan.md`](./plan.md)

Esquemas dos dados que fluem pelo pipeline após esta feature. A ordem das
colunas de entrada do modelo é regida pelo contrato de features
([`contracts/feature-contract.md`](./contracts/feature-contract.md),
código em `src/data/feature_contract.py`).

---

## 1. Eventos com features causais (`data/processed/{train,val,test}.parquet`)

Uma linha por evento. Produzido por `src/features/build_features.py`
(estágio `feature_eng`): features causais **antes** do split; split
cronológico por timestamp.

| Coluna | Tipo | Semântica (as-of: só eventos ANTERIORES à linha) |
|--------|------|--------------------------------------------------|
| `timestamp` | datetime64[ns] | Instante do evento |
| `visitorid` / `itemid` | int64 | Ids originais do RetailRocket |
| `user_idx` / `item_idx` | int64 | Ids contíguos (`pd.factorize`, começam em 0) |
| `event` | str | view / addtocart / transaction |
| `weight` | int64 | Peso do evento (1/3/5) — definido em `labeling.py` |
| `hour`, `day_of_week` | int32 | Contexto temporal da linha |
| `frequency` | int64 | Nº de eventos anteriores do usuário |
| `engagement_score` | int64 | Soma dos pesos dos eventos anteriores do usuário |
| `recency_days` | float64 | Dias desde o evento anterior do usuário (−1 = primeiro) |
| `view_count` | int64 | Views anteriores do item |
| `cat_idx` | int64 | Categoria do item (contíguo; −1 = sem categoria) |

**Regra split→agregação**: as agregações são causais por construção
(cada linha só enxerga o passado), então o split pode ser aplicado depois
sem vazamento. Nada é "fitado" neste estágio; o scaler é ajustado apenas
no treino, dentro do modelo (D5).

## 2. Dataset rotulado (em memória, `src/data/labeling.py`)

`build_labeled_dataset(events, history) → FEATURE_COLS + label`

- **Positivos** (`label=1.0`): linhas addtocart/transaction de `events`,
  com suas features causais.
- **Negativos** (`label=0.0`): por positivo, `num_negatives` (4) itens
  amostrados de `history` por popularidade^0.75, rejeitando itens já
  vistos pelo usuário (em `history ∪ events`). Features de usuário/contexto
  copiadas do positivo; `view_count` do item calculado *as-of* o timestamp
  do positivo.
- `history` = treino quando se rotula val/test (popularidade, views e
  catálogo vêm só do treino — FR-003).

## 3. Tensores do NCF (`src/data/dataset.py`)

`RetailRocketDataset` → tupla por amostra:

| Tensor | dtype | Conteúdo |
|--------|-------|----------|
| `user_idx` | long | Índice de embedding do usuário (roteado p/ unknown se ∉ treino) |
| `item_idx` | long | Índice de embedding do item |
| `cat_idx` | long | Índice de embedding da categoria |
| `x_cont` | float32 | `CONT_COLS` escaladas pelo StandardScaler do modelo |
| `y` | float32 | Rótulo 0/1 |

Tabelas de embedding têm tamanho `n+1`; o índice `n` é o "unknown" (D4).

## 4. Candidatos da avaliação Top-K (em memória, `src/evaluation/ranking.py`)

Por usuário do teste: todos os seus positivos (relevância forte, ou ampla
com teto de 20 mais recentes) + 100 negativos uniformes não vistos do
catálogo do histórico. Mesma semente → mesmos candidatos para modelo e
baseline de popularidade (AC-1).

## 5. Modelo serializado (`models/artifacts/<run_id>/model.pkl`)

`NCFRecommender` auto-contido (D5): rede + `StandardScaler` ajustado no
treino + metadados (`n_users`, `n_items`, `n_categories`,
`item_categories`, máscaras `known_users`/`known_items`,
`contract_version`). Serving e avaliação entregam a matriz crua na ordem
do contrato; o modelo aplica roteamento de ids e escala internamente.
