# Pipeline de Pré-Processamento e Feature Engineering

## Visão Geral

O pipeline transforma dados brutos do RetailRocket em features prontas para treinamento em quatro estágios DVC encadeados.

```
data/raw/events.csv
        │
        ▼  [preprocess]
data/interim/data_clean.parquet
        │
        ▼  [feature_eng]
data/processed/{train,val,test}.parquet
        │
        ▼  [train]
models/artifacts/
        │
        ▼  [evaluate]
metrics/
```

Para reproduzir do zero:

```bash
dvc repro
```

---

## Estágio 1 — Pré-Processamento (`preprocess`)

**Entrada:** `data/raw/events.csv`  
**Saída:** `data/interim/data_clean.parquet`  
**Módulo:** `src/data/preprocess.py` + `src/data/preprocessor.py`

### Operações aplicadas

| Ordem | Operação | Implementação |
|-------|----------|---------------|
| 1 | Filtra usuários com < `min_interactions` eventos | `RetailRocketPreprocessor._filter_users()` |
| 2 | Converte `visitorid` → `user_idx` (int contíguo, base 0) | `RetailRocketPreprocessor._encode_ids()` |
| 3 | Converte `itemid` → `item_idx` (int contíguo, base 0) | `RetailRocketPreprocessor._encode_ids()` |
| 4 | Ordena por `timestamp` | `preprocessor.preprocess()` |

### Parâmetros (em `configs/params.yaml`)

```yaml
preprocess:
  random_state: 42
```

`min_interactions` é fixado em 5 conforme análise do EDA (ver [eda.md](eda.md#decisões-para-o-pré-processamento)).

### Pattern utilizado

`RetailRocketPreprocessor` implementa a interface `PreprocessStrategy` (Strategy Pattern). Para trocar a estratégia:

```python
from src.data.preprocessor import RetailRocketPreprocessor

preprocessor = RetailRocketPreprocessor(min_interactions=5)
df_clean = preprocessor.preprocess(df_raw)
```

### Schema de saída

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `timestamp` | datetime64 | Data/hora do evento |
| `visitorid` | int64 | ID original do usuário (mantido para rastreabilidade) |
| `event` | str | Tipo: `view`, `addtocart`, `transaction` |
| `itemid` | int64 | ID original do item (mantido para rastreabilidade) |
| `user_idx` | int64 | Índice contíguo do usuário (0-based) |
| `item_idx` | int64 | Índice contíguo do item (0-based) |

---

## Estágio 2 — Feature Engineering (`feature_eng`)

**Entrada:** `data/interim/data_clean.parquet`  
**Saída:** `data/processed/train.parquet`, `val.parquet`, `test.parquet`  
**Módulos:** `src/data/feature_engineering.py` (lógica) + `src/features/build_features.py` (estágio DVC)

### Funções de feature engineering

Todas as funções em `src/data/feature_engineering.py` são **puras** (sem efeitos colaterais, sem I/O).

#### `add_event_weights(df)`

Adiciona coluna `weight` com o peso implícito de cada tipo de evento.

| Evento | Peso | Justificativa |
|--------|------|---------------|
| `view` | 1 | Intenção baixa — navegação passiva |
| `addtocart` | 3 | Intenção média — consideração ativa |
| `transaction` | 5 | Intenção máxima — compra efetivada |

#### `add_temporal_features(df)`

Extrai features de tempo do timestamp:

| Coluna nova | Valores | Uso no modelo |
|-------------|---------|---------------|
| `hour` | 0–23 | Padrão horário de compra |
| `day_of_week` | 0 (seg) – 6 (dom) | Padrão semanal |

#### `compute_user_features(df, reference_date=None)`

Agrega comportamento histórico por `user_idx`. `reference_date` padrão = dia seguinte ao último evento (normalizável).

| Coluna | Descrição | Fórmula |
|--------|-----------|---------|
| `frequency` | Total de interações | `COUNT(*)` por `user_idx` |
| `recency_days` | Dias desde último evento | `(ref_date - last_event).days` com normalização por dia |
| `engagement_score` | Engajamento ponderado | `SUM(weight)` por `user_idx` |

#### `compute_item_features(df)`

Agrega popularidade por `item_idx`.

| Coluna | Descrição | Fórmula |
|--------|-----------|---------|
| `view_count` | Total de views | `COUNT(event == 'view')` por `item_idx` |
| `popularity_tier` | Tier de popularidade | Baseado em quantis de `view_count` |

Tiers de popularidade:

| Tier | Critério |
|------|----------|
| `top_tier` | `view_count` > P90 |
| `mid_tier` | P50 < `view_count` ≤ P90 |
| `long_tail` | `view_count` ≤ P50 |

#### `build_interaction_features(df)`

Orquestra todas as funções acima e faz merge no nível de evento. Retorna um DataFrame com N linhas (uma por interação) e todas as features de usuário, item e temporais.

#### `chronological_split(df, val_size, test_size)`

Divide eventos por ordem de `timestamp` — sem aleatoriedade.

```
|←─── train (~70%) ───→|←─ val (~10%) ─→|←─ test (~20%) ─→|
                        ↑               ↑
                   timestamp           timestamp
```

**Sem data leakage:** eventos de teste são sempre posteriores a todos os eventos de treino e validação.

### Parâmetros (em `configs/params.yaml`)

```yaml
preprocess:
  test_size: 0.2
  val_size: 0.1
  random_state: 42  # ignorado no split cronológico
```

---

## Dataset PyTorch (`src/data/dataset.py`)

`RetailRocketDataset` encapsula os splits processados para uso no DataLoader:

```python
from src.data.dataset import RetailRocketDataset
from torch.utils.data import DataLoader

feature_cols = ["frequency", "recency_days", "engagement_score",
                "view_count", "hour", "day_of_week"]

dataset = RetailRocketDataset(train_df, feature_cols, target_col="weight")
loader = DataLoader(dataset, batch_size=256, shuffle=True)
```

---

## Decisões de Design

| Decisão | Motivo |
|---------|--------|
| Funções puras em `feature_engineering.py` | Testabilidade sem disco; composição simples |
| Split cronológico em vez de aleatório | Evita data leakage em séries temporais |
| `user_idx` / `item_idx` contíguos base-0 | Compatíveis diretamente com `nn.Embedding` do PyTorch |
| Strategy Pattern no preprocessador | Troca de dataset sem alterar pipeline DVC |
| Pesos view=1, addtocart=3, transaction=5 | Hierarquia de intenção confirmada pelo EDA |
