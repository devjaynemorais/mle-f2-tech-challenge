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
| Testes | pytest + pytest-cov (57 testes, 100% sem I/O de disco) |

## Estrutura do Projeto

```
mle-f2-tech-challenge/
├── config/               # config.yaml (paths de dados, MLflow e modelo de produção)
├── data/
│   ├── raw/              # Dados brutos (rastreados pelo DVC)
│   ├── interim/          # Saída do estágio preprocess (data_clean.parquet)
│   ├── processed/        # Splits train/val/test (saída do feature_eng)
│   └── external/         # Dados externos de referência
├── docs/                 # eda.md, preprocessing.md, tests.md, make.md,
│                         # exploration_doc.md, model_card.md
├── metrics/              # Métricas DVC (JSON) e plots
├── models/
│   ├── artifacts/        # Artefatos do modelo por run MLflow
│   └── production/       # Modelo promovido a Production
├── notebooks/            # EDA e exploração de arquiteturas de modelos
├── scripts/              # download_dataset.py, validate_env.py
├── src/
│   ├── config/           # settings.py (Pydantic Settings + .env)
│   ├── data/             # preprocess.py, preprocessor.py,
│   │                     # feature_engineering.py, dataset.py, make_dataset.py
│   ├── features/         # build_features.py (estágio DVC feature_eng)
│   ├── models/           # factory.py, base.py, mlp.py, baselines.py, registry.py
│   ├── training/         # trainer.py
│   ├── evaluation/       # evaluate.py
│   ├── serving/          # api.py, store.py, model_loader.py, recommender.py
│   └── utils/            # seed.py, eda.py, plots.py, mlflow_tracking.py,
│                         # logging_config.py
├── tests/                # test_preprocess.py, test_feature_engineering.py,
│                         # test_smoke.py, test_registry.py, test_serving.py
├── dvc.yaml              # Definição dos 5 estágios do pipeline
├── params.yaml           # Hiperparâmetros versionados
├── pyproject.toml        # Dependências e configuração de ferramentas
├── poetry.lock
├── Dockerfile            # 4 stages: builder, runtime, mlflow-server, api
└── docker-compose.yml
```

## Início Rápido

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
data/raw/events.csv
        │
        ▼  preprocess   →  data/interim/data_clean.parquet
        ▼  feature_eng  →  data/processed/{train,val,test}.parquet
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
make feature-eng
make train
make evaluate
```

## Desenvolvimento

```bash
make env          # primeira vez, ou após alterar pyproject.toml
make validate-env # checar variáveis de ambiente e dependências
make lint         # ruff check + verificação de formatação
make format       # corrigir lint e formatação automaticamente
make test         # pytest tests/ -v  (57 testes)
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

Detalhes completos em [`docs/preprocessing.md`](docs/preprocessing.md).

**Estágio `preprocess`:** filtra usuários com < 5 interações, encoda `visitorid` → `user_idx` e `itemid` → `item_idx` (base-0, contíguos), ordena por timestamp.

**Estágio `feature_eng`:** gera pesos implícitos (view=1, addtocart=3, transaction=5), features temporais (`hour`, `day_of_week`), features de usuário (`frequency`, `recency_days`, `engagement_score`) e de item (`view_count`, `popularity_tier`). Split cronológico sem data leakage: train (~70%) → val (~10%) → test (~20%).

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

Avaliação Top-K aplicada a todos os modelos:

| Métrica | Descrição |
|---------|-----------|
| Hit Rate@K | ≥ 1 item relevante entre os Top-K recomendados |
| Precision@K | Proporção de itens relevantes entre os Top-K |
| Recall@K | Proporção de itens relevantes do usuário recuperados |
| NDCG@K | Qualidade do ranking — penaliza acertos em posições baixas |

## Design Patterns

- **Factory** (`src/models/factory.py`): `ModelFactory.create("mlp")` instancia qualquer recomendador registrado via `@ModelFactory.register`. Modelo configurado em `params.yaml`.
- **Strategy** (`src/data/preprocessor.py`): `RetailRocketPreprocessor` implementa `PreprocessStrategy`, permitindo trocar a lógica de pré-processamento sem alterar o pipeline DVC.

## Testes

Detalhes em [`docs/tests.md`](docs/tests.md).

| Arquivo | Escopo | Testes |
|---------|--------|--------|
| `tests/test_preprocess.py` | DefaultPreprocessor e RetailRocketPreprocessor | 11 |
| `tests/test_feature_engineering.py` | Todas as funções de feature engineering e split | 23 |
| `tests/test_smoke.py` | ModelFactory e MLP (fit + predict) | 4 |
| `tests/test_registry.py` | Registro e promoção no MLflow Registry | 3 |
| `tests/test_serving.py` | FeatureStore, model loader, RecommendationService e endpoints | 16 |
| **Total** | | **57** |

Todos os testes usam dados sintéticos em memória — sem leitura de `data/` nem
dependência de servidor MLflow.

## Parâmetros

`params.yaml` — versionado com DVC:

```yaml
train:
  model_type: mlp        # mlp | logistic | dummy
  epochs: 50
  batch_size: 256
  learning_rate: 0.001
  early_stopping_patience: 5
  random_state: 42

mlflow:
  experiment_name: recommendation_system
  run_name: baseline
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
