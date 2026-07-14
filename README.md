# mle-f2-tech-challenge

Sistema de recomendação de produtos para e-commerce — FIAP MLE Fase 2.

## Problema

Uma empresa de e-commerce precisa de um sistema de recomendação de produtos baseado no comportamento de navegação dos usuários. O modelo central é uma rede neural (MLP ou embedding-based) treinada com PyTorch, com pipeline completo containerizado em Docker, dados versionados com DVC e experimentos rastreados no MLflow.

## Stack

| Camada | Ferramenta |
|--------|------------|
| Modelo | PyTorch (MLP + embeddings), Scikit-Learn (baselines) |
| Rastreamento de experimentos | MLflow ≥ 3.0 |
| Versionamento de dados | DVC |
| Gerenciamento de dependências | Poetry 1.8.3 + `pyproject.toml` |
| Linting / formatação | ruff + mypy + pre-commit |
| Containerização | Docker multi-stage + docker-compose |
| Configuração | Pydantic Settings + `params.yaml` + `.env` |
| Testes | pytest + pytest-cov (79 testes, 100% sem I/O de disco) |

## Estrutura do Projeto

```
mle-f2-tech-challenge/
├── config/               # config.yaml (paths de dados, MLflow e modelo de produção)
├── data/
│   ├── raw/              # Dados brutos (rastreados pelo DVC)
│   ├── interim/          # Saída do estágio preprocess (data_clean.parquet)
│   ├── content/          # Saída do estágio content (item_categories.parquet)
│   ├── processed/        # Splits train/val/test (saída do feature_eng)
│   └── external/         # Dados externos de referência
├── docs/                 # how_to_run.md, eda.md, preprocessing.md, tests.md,
│                         # make.md, exploration_doc.md, model_card.md,
│                         # feature_selection.md, architecture.md
├── metrics/              # Métricas DVC (JSON) e plots
├── models/
│   ├── artifacts/        # Artefatos do modelo por run MLflow
│   └── production/       # Modelo promovido a Production
├── notebooks/            # EDA e exploração de arquiteturas de modelos
├── scripts/              # download_dataset.py, validate_env.py
├── src/
│   ├── config/           # settings.py (Pydantic Settings + .env)
│   ├── data/             # preprocess.py, preprocessor.py, content_etl.py,
│   │                     # feature_engineering.py, feature_contract.py,
│   │                     # labeling.py, dataset.py, make_dataset.py
│   ├── features/         # build_features.py (estágio DVC feature_eng)
│   ├── models/           # factory.py, base.py, mlp.py (NCF), baselines.py,
│   │                     # registry.py
│   ├── training/         # trainer.py
│   ├── evaluation/       # evaluate.py, ranking.py, scorers.py
│   ├── serving/          # api.py, store.py, model_loader.py, recommender.py
│   └── utils/            # seed.py, eda.py, plots.py, mlflow_tracking.py,
│                         # logging_config.py
├── tests/                # 79 testes (dados 100% sintéticos) — ver seção Testes
├── specs/                # Specs de features (001-recommender-quality)
├── dvc.yaml              # Definição dos 6 estágios do pipeline
├── params.yaml           # Hiperparâmetros versionados
├── pyproject.toml        # Dependências e configuração de ferramentas
├── poetry.lock
├── Dockerfile            # 4 stages: builder, runtime, mlflow-server, api
└── docker-compose.yml
```

## Início Rápido

> Guia completo de execução (passo a passo, Windows, troubleshooting):
> [`docs/how_to_run.md`](docs/how_to_run.md). Para validar os critérios de
> aceite da spec do recomendador:
> [`specs/001-recommender-quality/quickstart.md`](specs/001-recommender-quality/quickstart.md).

**Pré-requisitos:** Python 3.11+, pip, Make, Docker

```bash
# 1. Instalar Poetry 1.8.3 e todas as dependências (prod + dev) em .venv/
make env

# 2. Configurar variáveis de ambiente
cp .env.example .env

# 3. Obter os dados
dvc pull
# ou: python scripts/download_dataset.py  (requer KAGGLE_USERNAME e KAGGLE_KEY no .env)

# 4. Subir o MLflow (o pipeline registra runs em http://localhost:5000).
#    Rode em um terminal separado e deixe aberto:
make mlflow
# Alternativa sem servidor local: use Docker → make compose-pipeline (sobe o MLflow por você)

# 5. Validar ambiente e rodar pipeline completo (em outro terminal)
make setup
```

> **Nota:** o estágio `train` falha com `WinError 10061 / Connection refused` se o
> MLflow não estiver rodando. Deixe `make mlflow` ativo antes de `make setup`.

## Pipeline DVC

```
data/raw/events.csv                     data/raw/item_properties_part*.csv
        │                                       │
        ▼  preprocess   →  data/interim/        ▼  content → data/content/
        │                  data_clean.parquet   │            item_categories.parquet
        └───────────────────┬───────────────────┘
        ▼  feature_eng  →  data/processed/{train,val,test}.parquet  (features causais)
        ▼  train        →  models/artifacts/ + metrics/train_metrics.json
        ▼  evaluate     →  metrics/eval_metrics.json + metrics/plots/
        ▼  promote      →  models/promoted_model.json (Registry: Staging → Production)
```

Reproduzir do zero:

```bash
dvc repro
```

Estágios individuais:

```bash
make preprocess
poetry run python -m src.data.content_etl   # estágio content (categoryid)
make feature-eng
make train
make evaluate
make promote
```

> **Windows/PowerShell:** rode o DVC via `poetry run dvc repro` (garante o
> Python da venv) e com `$env:PYTHONUTF8 = "1"` (evita `UnicodeEncodeError`
> do MLflow no console cp1252). Detalhes em [`docs/how_to_run.md`](docs/how_to_run.md).

## Desenvolvimento

```bash
make env          # primeira vez, ou após alterar pyproject.toml
make validate-env # checar variáveis de ambiente e dependências
make lint         # ruff check + verificação de formatação
make format       # corrigir lint e formatação automaticamente
make test         # pytest tests/ -v  (79 testes)
make test-cov     # pytest com relatório HTML em htmlcov/
```

## Serviços Locais

```bash
make mlflow   # MLflow UI em http://localhost:5000
make api      # FastAPI em http://localhost:8000 com hot-reload
```

> **Portas ocupadas?** Ambos os alvos aceitam override de porta. Se a 5000 estiver
> em uso, rode `make mlflow MLFLOW_PORT=5001` (e ajuste `MLFLOW_TRACKING_URI` no
> `.env`). Se a 8000 estiver em uso (ex.: Docker Desktop), rode
> `make api API_PORT=8001` e teste em `http://localhost:8001`.

## API de Serving

FastAPI em `src/serving/`, servida por `uvicorn src.serving.api:app`. No startup,
carrega o modelo de produção e um feature store em memória (uma única vez).

**Carregamento do modelo** (`model_loader.py`): tenta o MLflow Registry
(`models:/retailrocket_recommender/Production`) primeiro; se o servidor MLflow
estiver inacessível, cai para o artefato local (`models/promoted_model.json` →
`models/artifacts/<run_id>/model.pkl`). Um pré-check de TCP (2 s) evita travar
quando não há servidor.

**Feature store** (`store.py`): lê `data/processed/` e monta lookups de features
por usuário, `view_count` por item, itens já vistos e ranking de popularidade.

| Endpoint | Descrição |
|----------|-----------|
| `GET /` | Metadados do serviço |
| `GET /health` | `{status, model_loaded, n_users, n_items}` |
| `GET /recommend?user_id=X&top_k=10` | Top-K itens rankeados (`top_k` entre 1 e 100) |

Para um usuário conhecido, pontua os itens candidatos não vistos com o modelo e
ranqueia (`strategy: "model"`). Para um usuário desconhecido (cold start), retorna
os itens mais populares (`strategy: "popularity"`).

```bash
curl "http://localhost:8000/recommend?user_id=11883&top_k=5"
```

```json
{
  "user_id": 11883,
  "strategy": "model",
  "count": 5,
  "recommendations": [
    { "item_idx": 18205, "score": 0.066388 }
  ]
}
```

## Docker

```bash
make compose-build   # builda as imagens
make compose-full    # sobe mlflow + train + api
make compose-down    # para e remove os containers
```

Serviços do `docker-compose.yml`:

| Serviço | Porta | Descrição |
|---------|-------|-----------|
| `mlflow` | 5000 | MLflow Tracking Server (SQLite backend) |
| `train` | — | Treino do modelo (aguarda mlflow estar saudável) |
| `evaluate` | — | Avaliação do modelo (profile `eval`) |
| `api` | 8000 | FastAPI serving endpoint |

## Dataset

**RetailRocket E-commerce Dataset** — interações de usuários em loja virtual.

| Arquivo | Descrição |
|---------|-----------|
| `events.csv` | ~2,8M interações usuário-item com timestamp e tipo de evento |
| `item_properties_part1/2.csv` | Propriedades dos itens ao longo do tempo |
| `category_tree.csv` | Hierarquia de categorias |

Principais características (ver [`docs/eda.md`](docs/eda.md)):

- ~1,4M usuários únicos · ~235K itens únicos · 4,5 meses de dados (Mai–Out 2015)
- Esparsidade da matriz usuário × item: **> 99,99%**
- Distribuição de interações segue **power-law**: mediana de ~2 interações por usuário
- Funil de conversão: view → addtocart (~4–5%) → transaction (~35–45%)

Download: [kaggle.com/datasets/retailrocket/ecommerce-dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

Após baixar, colocar os arquivos em `data/raw/` e rodar `dvc repro`.

## Pré-Processamento e Features

Detalhes completos em [`docs/preprocessing.md`](docs/preprocessing.md). Racional de escolha de cada feature, gaps e candidatas a adicionar em [`docs/feature_selection.md`](docs/feature_selection.md).

**Estágio `preprocess`:** filtra usuários com < 5 interações, encoda `visitorid` → `user_idx` e `itemid` → `item_idx` (base-0, contíguos), ordena por timestamp.

**Estágio `content`:** extrai o `categoryid` mais recente de cada item dos `item_properties_part*.csv` (leitura em chunks) — vira embedding de categoria no modelo (cold-start de conteúdo).

**Estágio `feature_eng`:** gera pesos implícitos (view=1, addtocart=3, transaction=5), features temporais (`hour`, `day_of_week`) e features **causais** (*as-of*, cada linha usa só eventos anteriores — sem vazamento temporal): usuário (`frequency`, `engagement_score`, `recency_days`) e item (`view_count`, `cat_idx`). Split cronológico: train (~70%) → val (~10%) → test (~20%).

**Labeling (no estágio `train`):** positivo = addtocart/transaction; negativos = pares (user, item) não interagidos, amostrados uniformemente a 8:1 — definição única em `src/data/labeling.py`.

## Modelos

A exploração foi conduzida em ordem crescente de complexidade — ver [`docs/exploration_doc.md`](docs/exploration_doc.md) e `notebooks/02_exploration_MLP_Architecture.ipynb`:

| Modelo | Tipo | Descrição |
|--------|------|-----------|
| Popularidade Ponderada | Baseline | Ranking global por score ponderado de eventos |
| Basket Analysis (Apriori) | Baseline | Regras de associação entre produtos coocorrentes |
| Item-Based CF | Baseline | Similaridade de cosseno via matriz usuário-item esparsa |
| Matrix Factorization (ALS) | Baseline | Fatoração com feedback implícito via `implicit` |
| Neural CF (NCF) | Neural | Embeddings user+item concatenados em MLP com BCEWithLogitsLoss |
| Two-Tower | Neural | Torres separadas para usuário e item; score por produto escalar |

O modelo principal do pipeline DVC é o **NCF (MLP)** configurado em `params.yaml`.

## Métricas de Avaliação

Avaliação Top-K por usuário (K = 10 e 20), sob relevância **forte**
(addtocart/transaction) e **ampla** (inclui view), sempre comparando o
modelo com o **baseline de popularidade** nos mesmos candidatos e
segmentando usuários **warm** (com histórico de treino) vs. **cold**:

| Métrica | Descrição |
|---------|-----------|
| Hit Rate@K | ≥ 1 item relevante entre os Top-K recomendados |
| Precision@K | Proporção de itens relevantes entre os Top-K |
| Recall@K | Proporção de itens relevantes do usuário recuperados |
| NDCG@K | Qualidade do ranking — penaliza acertos em posições baixas |

Além disso, o teste rotulado reporta ROC-AUC, Average Precision, F1,
Precisão e Recall. Resultados e limitações: [`docs/model_card.md`](docs/model_card.md).

## Design Patterns

- **Factory** (`src/models/factory.py`): `ModelFactory.create("ncf")` instancia qualquer recomendador registrado via `@ModelFactory.register`. Modelo configurado em `params.yaml`.
- **Strategy** (`src/data/preprocessor.py`): `RetailRocketPreprocessor` implementa `PreprocessStrategy`, permitindo trocar a lógica de pré-processamento sem alterar o pipeline DVC.

## Testes

Detalhes em [`docs/tests.md`](docs/tests.md).

| Arquivo | Escopo | Testes |
|---------|--------|--------|
| `tests/test_preprocess.py` | DefaultPreprocessor e RetailRocketPreprocessor | 11 |
| `tests/test_feature_engineering.py` | Features causais (as-of), ausência de vazamento, split | 12 |
| `tests/test_labeling.py` | Rótulo único, negative sampling, determinismo | 11 |
| `tests/test_ranking.py` | Métricas Top-K e protocolo por usuário | 12 |
| `tests/test_ncf.py` | NCF: shapes, roteamento unknown, sanidade de aprendizado | 7 |
| `tests/test_contract.py` | Contrato de features treino ↔ serving | 3 |
| `tests/test_smoke.py` | ModelFactory e NCF (fit + predict) | 4 |
| `tests/test_registry.py` | Registro e promoção no MLflow Registry | 3 |
| `tests/test_serving.py` | FeatureStore, model loader, RecommendationService e endpoints | 16 |
| **Total** | | **79** |

Todos os testes usam dados sintéticos em memória — sem leitura de `data/` nem
dependência de servidor MLflow.

## Parâmetros

`params.yaml` — versionado com DVC (valores completos no arquivo):

```yaml
labeling:
  num_negatives: 8          # negativos por positivo (amostragem uniforme)
  popularity_alpha: 0.0

train:
  model_type: ncf           # ncf | logistic | dummy
  epochs: 40
  embedding_dim: 64
  hidden_dims: [128, 64]
  unknown_dropout: 0.1      # treina o embedding "unknown" (cold-start)
  weight_decay: 0.0001
  early_stopping_patience: 5

eval:
  k_values: [10, 20]           # métricas Top-K
  num_candidate_negatives: 100 # negativos por usuário na avaliação

registry:
  metric: val_ndcg_at_20    # métrica de validação usada na promoção
```

## Critérios de Avaliação

| Critério | Peso | Descrição |
|----------|------|-----------|
| Clean code e estrutura | 20% | SOLID, naming, type hints, design patterns, linting |
| Reprodutibilidade | 15% | Poetry, lock file, .env, instalação limpa |
| Docker | 15% | Multi-stage, imagem otimizada, compose funcional |
| DVC + Pipeline | 15% | Dataset versionado, pipeline ≥ 4 stages, dvc repro funcional |
| Rede neural (PyTorch) | 15% | MLP funcional, early stopping, comparação com baselines |
| MLflow + Registry | 10% | ≥ 3 runs rastreados, modelo promovido a Production |
| Vídeo STAR | 10% | Clareza, cobertura dos 4 elementos, ≤ 5 min |
| Bônus: deploy em nuvem | +5% | Container acessível via URL pública |
