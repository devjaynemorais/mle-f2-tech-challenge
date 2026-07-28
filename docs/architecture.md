# Documentação Técnica — Arquitetura do Sistema

Referência técnica consolidada do sistema de recomendação RetailRocket. Cobre
os cinco pilares do projeto: **pipeline**, **testes**, **ambiente**, **MLflow** e
**API de serving**. Complementa os documentos temáticos existentes
([`eda.md`](eda.md), [`preprocessing.md`](preprocessing.md),
[`exploration_doc.md`](exploration_doc.md), [`model_card.md`](model_card.md),
[`make.md`](make.md), [`tests.md`](tests.md)).

---

## Visão Geral

```
                    ┌──────────────────────────────────────────────┐
                    │                  params.yaml                  │
                    │        (hiperparâmetros versionados)          │
                    └──────────────────────────────────────────────┘
                                        │
 data/raw/events.csv                    │ lê parâmetros por estágio
        │                               ▼
        │  ┌───────────┐   ┌────────────┐   ┌────────┐   ┌──────────┐   ┌─────────┐
        └─▶│preprocess │──▶│feature_eng │──▶│ train  │──▶│ evaluate │──▶│ promote │
           └───────────┘ ┌▶└────────────┘   └────────┘   └──────────┘   └─────────┘
 data/raw/               │      │              │              │             │
 item_properties_*.csv   │ data/processed/  models/       metrics/       models/
        │  ┌───────────┐ │ train/val/test   artifacts/    eval_metrics   promoted_
        └─▶│ content   │─┘ .parquet         + MLflow run  + plots         model.json
           └───────────┘
                │
        data/content/
        item_categories
        .parquet
                                               │              │             │
                                               ▼              ▼             ▼
                                        ┌──────────────────────────────────────┐
                                        │           MLflow Tracking            │
                                        │   runs · métricas · artefatos ·      │
                                        │   Model Registry (Staging→Production)│
                                        └──────────────────────────────────────┘
                                                          │
                                                          ▼
                                        ┌──────────────────────────────────────┐
                                        │      FastAPI (src/serving/api.py)     │
                                        │  carrega modelo Production + store    │
                                        │  GET /  ·  /health  ·  /recommend     │
                                        └──────────────────────────────────────┘
```

O orquestrador é o **DVC** (`dvc.yaml`): cada estágio declara `deps`, `params`,
`outs`/`metrics`/`plots`, e o DVC só reexecuta o que ficou desatualizado.
`params.yaml` é a única fonte de hiperparâmetros; `.env` + Pydantic Settings
(`src/config/settings.py`) resolvem infraestrutura (URI do MLflow, caminhos,
portas). Todos os estágios são módulos `python -m src.<pacote>.<módulo>`.

---

## 1. Pipeline

Definido em [`dvc.yaml`](../dvc.yaml). Reproduzir tudo: `dvc repro` (ou
`make setup`, que roda `validate-env` antes). Estágios individuais têm alvos
Make dedicados (`make preprocess`, `make feature-eng`, `make train`,
`make evaluate`, `make promote`).

| Estágio | Comando | Entradas (`deps`) | Saídas (`outs`/`metrics`) | Params |
|---------|---------|-------------------|---------------------------|--------|
| `preprocess`  | `python -m src.data.preprocess` | `src/data/preprocess{,or}.py`, `data/raw/events.csv` | `data/interim/` | `preprocess` |
| `content`     | `python -m src.data.content_etl` | `src/data/content_etl.py`, `data/raw/item_properties_part*.csv` | `data/content/` | — |
| `feature_eng` | `python -m src.features.build_features` | `src/features/build_features.py`, `src/data/feature_engineering.py`, `data/interim/`, `data/content/` | `data/processed/` | `preprocess`, `feature_eng` |
| `train`       | `python -m src.training.trainer` | `src/training/trainer.py`, `src/models/`, `src/data/labeling.py`, `data/processed/` | `models/artifacts/`, `metrics/train_metrics.json` | `train`, `labeling`, `eval` |
| `evaluate`    | `python -m src.evaluation.evaluate` | `src/evaluation/*.py`, `src/data/labeling.py`, `models/artifacts/`, `data/processed/` | `metrics/eval_metrics.json`, `metrics/plots/` | `labeling`, `eval` |
| `promote`     | `python -m src.models.registry` | `src/models/registry.py`, `src/utils/mlflow_tracking.py`, `metrics/*.json` | `models/promoted_model.json` | `registry`, `mlflow` |

### 1.1 `preprocess` — dados brutos → intermediários limpos

Fonte: [`src/data/preprocess.py`](../src/data/preprocess.py) +
[`src/data/preprocessor.py`](../src/data/preprocessor.py).

1. Lê `data/raw/events.csv` e converte `timestamp` de epoch-ms para `datetime`.
2. Aplica `RetailRocketPreprocessor` (Strategy pattern — implementa
   `PreprocessStrategy`):
   - **Filtra usuários frios**: remove `visitorid` com menos de
     `min_interactions` (`params.yaml`, hoje **5**) eventos.
   - **Encoda IDs**: `pd.factorize` mapeia `visitorid → user_idx` e
     `itemid → item_idx` como inteiros contíguos base-0.
   - **Ordena** por `timestamp`.
3. Grava `data/interim/data_clean.parquet`.

Se `data/raw/events.csv` não existir, o estágio loga um aviso e retorna sem erro
(permite `dvc repro` parcial em clones sem os dados brutos).

### 1.2 `content` + `feature_eng` — conteúdo e features causais + splits

Fontes: [`src/data/content_etl.py`](../src/data/content_etl.py),
[`src/features/build_features.py`](../src/features/build_features.py) e
[`src/data/feature_engineering.py`](../src/data/feature_engineering.py). Todas as
funções de feature são **puras** (`DataFrame → DataFrame`, sem I/O).

O estágio `content` lê `item_properties_part*.csv` em chunks, filtra
`property == "categoryid"` e mantém o valor mais recente por item →
`data/content/item_categories.parquet` (usado pelo embedding de categoria
do NCF — cold-start de conteúdo).

`build_causal_features` compõe quatro grupos de features **causais (as-of)**
por linha de evento — cada linha enxerga apenas eventos ANTERIORES a ela
(FR-003; elimina o vazamento temporal da agregação global antiga):

| Grupo | Colunas | Regra (as-of) |
|-------|---------|---------------|
| Evento | `weight` | `view=1`, `addtocart=3`, `transaction=5` (`EVENT_WEIGHTS`, definido em `src/data/labeling.py`) |
| Temporal | `hour`, `day_of_week` | extraídas do `timestamp` (0=segunda … 6=domingo) |
| Usuário | `frequency`, `engagement_score`, `recency_days` | nº de eventos anteriores; soma dos pesos anteriores; dias desde o evento anterior (−1 = primeiro evento) |
| Item | `view_count`, `cat_idx` | views anteriores do item; categoria contígua (−1 = sem categoria) |

`chronological_split` ordena por `timestamp` e corta por posição (sem embaralhar),
evitando *data leakage* temporal:

```
train = [0 : n - n_test - n_val]     (~70%)
val   = [n - n_test - n_val : n - n_test]  (~10%)
test  = [n - n_test :]               (~20%)
```

`test_size` e `val_size` vêm de `params.preprocess`. O `random_state` é aceito na
assinatura mas ignorado — o split é determinístico por tempo. Saídas:
`data/processed/{train,val,test}.parquet`.

### 1.3 `train` — labeling + NCF + loga no MLflow

Fonte: [`src/training/trainer.py`](../src/training/trainer.py) +
[`src/data/labeling.py`](../src/data/labeling.py).

- **Contrato de features** (`src/data/feature_contract.py`, 8 colunas):
  `user_idx, item_idx | hour, day_of_week, frequency, engagement_score, recency_days, view_count`
  — fonte única para treino, avaliação e serving (FR-007).
- **Rótulo (FR-001/002, definição única em `labeling.py`)**: positivo =
  `addtocart`/`transaction`; negativo = pares (user, item) **não interagidos**,
  amostrados por popularidade^0.75 na razão 4:1, com `view_count` *as-of* o
  timestamp do positivo. Val é rotulada com estatísticas do TREINO.
- **Modelo** criado por `ModelFactory.create` com `params.train`. Padrão:
  `ncf` — embeddings de usuário/item/categoria (último índice = "unknown",
  D4) ⊕ contínuas escaladas → MLP → logit.
- **Early stopping por val-AUC** (FR-010), restaurando o melhor estado.
- Loga `val_auc` **e** `val_ndcg_at_20` (métrica de promoção); salva
  `models/artifacts/<run_id>/model.pkl` **auto-contido** (rede + scaler +
  vocabulários/máscaras — D5) e `metrics/train_metrics.json`.

### 1.4 `evaluate` — classificação + ranking Top-K no teste

Fonte: [`src/evaluation/evaluate.py`](../src/evaluation/evaluate.py) +
[`ranking.py`](../src/evaluation/ranking.py) / [`scorers.py`](../src/evaluation/scorers.py).

- **Classificação** sobre o teste rotulado (positivos reais + negativos
  amostrados — a mesma tarefa do treino): `roc_auc`, `average_precision`,
  `f1`, `precision`, `recall` (threshold `0.5`).
- **Ranking Top-K por usuário** (FR-005/005a/006, D3): todos os positivos
  de teste do usuário + 100 negativos não vistos numa mesma lista;
  NDCG/Recall/Precision/HitRate @{10,20}; relevância **forte**
  (addtocart/transaction) e **ampla** (inclui view, teto de 20 positivos);
  modelo vs. **baseline de popularidade** nos MESMOS candidatos; métricas
  segmentadas **warm** (usuário no treino) vs. **cold** (FR-011).
- `_save_plots` grava `metrics/plots/{roc,pr}_curve.json`; métricas em
  `metrics/eval_metrics.json` (aninhado) e achatadas num run MLflow
  (`run_name=f"evaluate-{model_type}"`, ex.: `evaluate-ncf`).

### 1.5 `promote` — Model Registry (Staging → Production)

Fonte: [`src/models/registry.py`](../src/models/registry.py) +
[`src/utils/mlflow_tracking.py`](../src/utils/mlflow_tracking.py).

1. `find_best_model_run(experiment, metric, ascending)` busca (via
   `MlflowClient.search_runs`) o melhor run por `val_ndcg_at_20` (métrica de
   VALIDAÇÃO de ranking — FR-010) **que tenha o artefato** `model/model.pkl`
   registrado — descartando runs de avaliação sem modelo.
2. `register_model` cria o modelo registrado (idempotente) e uma nova versão a
   partir de `runs:/<run_id>/model/model.pkl`, transicionando para `Staging`
   (`archive_existing_versions=True`).
3. `promote_model` move a versão para `Production` (também arquivando as
   anteriores).
4. Grava `models/promoted_model.json` com `{model_name, version, stage, run_id,
   metric, value}` — o vínculo local que a API usa como *fallback* offline.

Parâmetros em `params.registry`: `model_name=retailrocket_recommender`,
`metric=val_ndcg_at_20`, `ascending=false`, `stage=Production`.

---

## 2. Design Patterns

| Pattern | Local | Papel |
|---------|-------|-------|
| **Factory** | `src/models/factory.py` | `ModelFactory.create("mlp")` instancia qualquer recomendador registrado via `@ModelFactory.register("nome")`. O tipo vem de `params.train.model_type`, então trocar o modelo é editar um YAML. Registro populado no import de `src/models/__init__.py`. |
| **Strategy** | `src/data/preprocessor.py` | `PreprocessStrategy` (ABC) com `RetailRocketPreprocessor` e `DefaultPreprocessor`. Permite trocar a lógica de limpeza sem tocar no estágio DVC. |
| **Template/ABC** | `src/models/base.py` | `RecommenderBase` define o contrato `fit` / `predict` / `predict_proba` que MLP e baselines respeitam — o que permite ao trainer, evaluate e serving tratarem qualquer modelo de forma uniforme. |

**Modelos registrados**: `ncf` (`src/models/mlp.py`), `logistic` e `dummy`
(`src/models/baselines.py`). O NCF concatena `nn.Embedding` de usuário, item e
categoria (tabelas `n+1`, último índice = unknown) às contínuas escaladas e
passa por um MLP (`Linear→ReLU→Dropout` × `[128, 64]` → 1 logit), treinado com
`BCEWithLogitsLoss`, `Adam` (+`weight_decay`) e **early stopping por
val-AUC** (restaura o melhor estado). `predict_proba` aplica `sigmoid`; o
scaler das contínuas vive dentro do modelo (D5). O baseline de
**popularidade** é avaliado direto no `evaluate` (mesmos candidatos do
modelo); os baselines de exploração adicionais (Apriori, Item-CF, ALS,
Two-Tower) estão nos notebooks — ver [`exploration_doc.md`](exploration_doc.md).

---

## 3. Testes

Fonte: [`tests/`](../tests/). Detalhe por-teste em [`tests.md`](tests.md).
Executar: `make test` (`poetry run pytest tests/ -v`) ou `make test-cov` (HTML em
`htmlcov/`). `pyproject.toml` já adiciona `--cov=src --cov-report=term-missing`.

| Arquivo | Escopo | Testes |
|---------|--------|--------|
| `test_preprocess.py` | `DefaultPreprocessor`, `RetailRocketPreprocessor` (filtro, encoding, ordenação) | 11 |
| `test_feature_engineering.py` | Features causais (as-of), ausência de vazamento (AC-3), `chronological_split` | 12 |
| `test_labeling.py` | Rótulo único, negative sampling 4:1, determinismo, view_count as-of | 11 |
| `test_ranking.py` | Métricas Top-K (casos à mão), protocolo por usuário, candidatos idênticos (AC-1) | 12 |
| `test_ncf.py` | Shapes, roteamento unknown (D4), overfit sintético, early stopping | 6 |
| `test_contract.py` | Contrato treino↔serving: mesmo vetor p/ mesmo (user, item) (FR-007) | 3 |
| `test_smoke.py` | `ModelFactory` e NCF (fit + predict em dados sintéticos) | 4 |
| `test_registry.py` | `find_best_model_run`, `register_model`, `promote_model` | 3 |
| `test_serving.py` | `FeatureStore`, `model_loader`, `RecommendationService`, endpoints | 16 |
| **Total** | | **78** |

**Princípios:**

- **Sem I/O de disco nem servidor.** Todos os testes constroem `pd.DataFrame`
  sintéticos em memória. Nenhum lê `data/` nem exige um MLflow rodando.
- **Registry isolado** (`test_registry.py`): usa um backend `sqlite:///` em
  `tmp_path` com `monkeypatch` sobre `settings.mlflow_tracking_uri` — roda em
  qualquer clone limpo.
- **API sem lifespan** (`test_serving.py`): instancia `TestClient(app)` **sem**
  `with`, o que **não** dispara o `lifespan` (portanto não carrega modelo do
  MLflow nem lê disco); o serviço é injetado manualmente em `api.state["service"]`.
- **Reachability sem rede real**: testa `_registry_reachable` contra uma porta
  fechada (falha rápida), um scheme não-HTTP (`file://` → `True`) e um socket
  local em escuta.
- **Contratos de erro da API**: `top_k` fora de `[1,100]` → **422**; serviço não
  carregado → **503**.

---

## 4. Ambiente & Reprodutibilidade

### 4.1 Dependências

- **Python** `^3.11`; **Poetry 1.8.3** (`make env` = `pip install poetry==1.8.3`
  + `poetry install --with dev`). `pyproject.toml` + `poetry.lock` fixam versões.
- Prod: `torch`, `scikit-learn`, `mlflow>=3.0`, `dvc>=3.50`, `pydantic-settings`,
  `pandas`, `numpy`, `pyarrow`, `fastapi`, `uvicorn[standard]`.
- Dev: `pytest`, `pytest-cov`, `ruff`, `mypy`, `pre-commit`, `httpx` (TestClient),
  `implicit`, `mlxtend`, `seaborn`, `jupyterlab` etc.

### 4.2 Configuração — Pydantic Settings

`src/config/settings.py` define `Settings(BaseSettings)`, que lê variáveis de
ambiente e o arquivo `.env` (`extra="ignore"`). Toda a infraestrutura passa por
aqui — o código nunca hardcoda URIs nem caminhos. Chaves principais e defaults:

| Campo | Default | Env var |
|-------|---------|---------|
| `mlflow_tracking_uri` | `http://localhost:5000` | `MLFLOW_TRACKING_URI` |
| `mlflow_experiment_name` | `recommendation_system` | `MLFLOW_EXPERIMENT_NAME` |
| `data_processed_path` | `data/processed` | `DATA_PROCESSED_PATH` |
| `model_artifacts_path` | `models/artifacts` | — |
| `production_model_name` | `retailrocket_recommender` | — |
| `production_stage` | `Production` | — |
| `serving_max_candidates` | `5000` | — |
| `api_host` / `api_port` | `0.0.0.0` / `8000` | `API_HOST` / `API_PORT` |
| `aws_*` | `None` / `us-east-1` | `AWS_*` (opcional, DVC remoto S3) |

Copiar `.env.example → .env` para começar. `scripts/validate_env.py`
(`make validate-env`) confere: pacotes obrigatórios importáveis, env vars
presentes (aviso, não erro — há defaults) e que `Settings()` carrega. Sai com
código 1 se pacotes faltarem ou o Settings falhar; `make setup` roda isso antes
do `dvc repro`.

### 4.3 Qualidade de código

- **ruff** (`make lint` / `make format`): `line-length=88`, regras
  `E,F,I,N,W,UP,B,ANN,D,C90`, docstrings estilo Google, complexidade McCabe ≤ 10.
  Ignora convenções ML (`X`, `y`, `X_train`) e docstrings redundantes.
- **mypy**: `python_version=3.11`, `ignore_missing_imports`, `warn_return_any`.
- **pre-commit** (`.pre-commit-config.yaml`) e versionamento de dados via **DVC**
  (`data/raw.dvc`, `dvc.lock`).

### 4.4 Docker

[`Dockerfile`](../Dockerfile) multi-stage (4 alvos) + `docker-compose.yml`:

| Stage | Base | Papel |
|-------|------|-------|
| `builder` | `python:3.11-slim` | `poetry install --only main` em `.venv` (in-project) |
| `runtime` | `python:3.11-slim` | copia `.venv` + `src/`+`config/`+`params.yaml`; roda train/evaluate/promote (`ENTRYPOINT ["python","-m"]`) |
| `mlflow-server` | `python:3.11-slim` | `mlflow==3.13.0`; serve tracking (ver §5) |
| `api` | `python:3.11-slim` | `uvicorn src.serving.api:app` na porta 8000 |

Serviços do compose: `mlflow` (5000, healthcheck em `/health`), `train`,
`evaluate` (profile `eval`), `promote` (profile `promote`) e `api` (8000). Os
jobs de pipeline usam `depends_on: mlflow condition: service_healthy` e recebem
`MLFLOW_TRACKING_URI=http://mlflow:5000` (nome de serviço na rede `ml-network`).
Volumes montam `./data`, `./models`, `./metrics` do host (a API monta `./data`
como *read-only*). Atalhos: `make compose-build`, `make compose-pipeline` (sobe
mlflow + train→evaluate→promote), `make compose-full`, `make compose-down`.

**Portas parametrizáveis**: `make mlflow MLFLOW_PORT=5001`,
`make api API_PORT=8001`, `MLFLOW_PORT=... docker compose up`.

---

## 5. MLflow

Duas responsabilidades: **Tracking** (runs, params, métricas, artefatos) e
**Model Registry** (versões + estágios Staging/Production).

### 5.1 Tracking server

| Contexto | Backend store | Artifact root |
|----------|---------------|---------------|
| Local (`make mlflow`) | `sqlite:///mlflow.db` | `./mlartifacts` |
| Docker (`mlflow-server`) | `sqlite:////mlartifacts/mlflow.db` | `/mlartifacts` (via `--serve-artifacts`) |

Clientes (train/evaluate/promote/serving) leem a URI de `settings.mlflow_tracking_uri`.

**Por que o servidor Docker usa `--serve-artifacts --artifacts-destination`:** os
containers de pipeline **não compartilham** o filesystem `/mlartifacts` do
servidor. Sem o proxy de artefatos, `log_artifact` gravaria num caminho local
inexistente no cliente e nada chegaria ao servidor. Com `--serve-artifacts`, o
upload passa **pelo** servidor MLflow.

**Por que `--allowed-hosts "*"`:** o MLflow 3.x valida o header `Host`
(proteção anti DNS-rebinding). Na rede do compose o cliente acessa via
`mlflow:5000`; liberar todos os hosts é aceitável em ambiente local — para deploy
público, restrinja a hosts específicos.

**Por que MLflow 3.x cliente ↔ servidor:** o cliente 3.x resolve URIs `runs:/`
via a API *logged-models*, ausente em servidores 2.x. `pyproject.toml` pede
`mlflow>=3.0` e a imagem fixa `mlflow==3.13.0` para casar.

### 5.2 Runs registrados

Por reprodução do pipeline: **train** (1 run com `val_auc`, params e artefato
`model/model.pkl`) + **evaluate** (1 run com as 5 métricas de teste). Rodar o
pipeline algumas vezes (ou variar hiperparâmetros) satisfaz o requisito de ≥ 3
runs rastreados. `find_best_model_run` compara todos por `val_auc`.

### 5.3 Model Registry

Helpers em `src/utils/mlflow_tracking.py`. Constante `MODEL_ARTIFACT_PATH =
"model/model.pkl"`. O nome registrado é `retailrocket_recommender`. Fluxo de
promoção detalhado na §1.5. `promote` grava `models/promoted_model.json` para que
a API funcione mesmo **sem** o servidor MLflow no ar.

---

## 6. API de Serving

Fonte: [`src/serving/`](../src/serving/). Rodar: `make api`
(`uvicorn src.serving.api:app --reload`) ou o serviço `api` do compose. Coleção
Postman em [`docs/RetailRocket Recommender API.postman_collection.json`](RetailRocket%20Recommender%20API.postman_collection.json).

### 6.1 Ciclo de vida (startup)

O `lifespan` (`api.py`) carrega **uma única vez**:

1. `load_production_model()` — o modelo de produção.
2. `FeatureStore.from_processed(data/processed/)` — lookups em memória.
3. Constrói `RecommendationService(model, store, serving_max_candidates)` e o
   guarda em `state["service"]`.

Se algo falhar, a exceção é logada e a API **sobe em modo degradado** —
`/health` reporta `{"status": "degraded", "model_loaded": false}` e `/recommend`
retorna **503**, em vez de o processo morrer.

### 6.2 Carregamento do modelo (Registry-first, fallback local)

`src/serving/model_loader.py`:

1. **Pré-check TCP (2 s)** — `_registry_reachable` abre um socket para o host:porta
   da URI. Se cair, falha rápido em vez de o cliente MLflow travar por minutos.
   (Schemes não-HTTP como `file://`/`db` retornam `True`.)
2. **Registry** — `get_latest_versions(name, [Production])`, baixa o artefato
   `model/model.pkl` do run e faz `pickle.load`.
3. **Fallback local** — em qualquer exceção, lê `models/promoted_model.json`,
   resolve `models/artifacts/<run_id>/model.pkl` e desserializa. Permite servir
   offline, sem MLflow.

### 6.3 Feature store

`src/serving/store.py` concatena `train/val/test.parquet` e monta:

- `user_idx → (frequency, engagement_score, recency_days)`
- `item_idx → view_count`
- `user_idx → {itens já vistos}`
- ranking de popularidade (`item_idx` ordenados por `view_count` desc)

Expõe `has_user`, `user_features`, `seen_items`, `candidate_items(max)`,
`view_counts`, `popular_items(k)`, `n_users`, `n_items`.

### 6.4 Lógica de recomendação

`src/serving/recommender.py` — `RecommendationService.recommend(user_id, top_k)`:

- **Usuário conhecido → `strategy: "model"`**: pega até `max_candidates` itens
  populares, **exclui os já vistos**, monta a matriz de features na **mesma ordem
  e semântica de `FEATURE_COLS`** (`hour`/`day_of_week` do relógio atual;
  `frequency/engagement_score/recency_days` do usuário; `view_count` por item),
  chama `predict_proba` e ordena decrescente, cortando em `top_k`.
- **Cold start ou candidatos esgotados → `strategy: "popularity"`**: se o usuário
  é desconhecido, *ou* já viu todos os candidatos, cai honestamente para os itens
  mais populares. A `strategy` reportada reflete o que de fato foi usado.

### 6.5 Endpoints

| Método | Rota | Resposta |
|--------|------|----------|
| `GET` | `/` | `{service, version}` |
| `GET` | `/health` | `{status, model_loaded[, n_users, n_items]}` |
| `GET` | `/recommend?user_id=X&top_k=K` | `{user_id, strategy, count, recommendations:[{item_idx, score}]}` |

`top_k` é validado em `[1, 100]` (`Query(10, ge=1, le=100)`) — fora disso a
FastAPI devolve **422**. Sem serviço carregado, `/recommend` devolve **503**.

```bash
curl "http://localhost:8000/recommend?user_id=11883&top_k=5"
```

```json
{
  "user_id": 11883,
  "strategy": "model",
  "count": 5,
  "recommendations": [ { "item_idx": 18205, "score": 0.066388 } ]
}
```

---

## Referência rápida de arquivos

| Área | Arquivos-chave |
|------|----------------|
| Orquestração | `dvc.yaml`, `params.yaml`, `Makefile` |
| Pipeline | `src/data/preprocess.py`, `src/data/content_etl.py`, `src/features/build_features.py`, `src/training/trainer.py`, `src/evaluation/evaluate.py`, `src/models/registry.py` |
| Features | `src/data/feature_engineering.py`, `src/data/feature_contract.py`, `src/data/labeling.py`, `src/data/preprocessor.py` |
| Modelos | `src/models/{base,factory,mlp,baselines}.py` |
| Avaliação | `src/evaluation/{evaluate,ranking,scorers}.py` |
| MLflow | `src/utils/mlflow_tracking.py`, `Dockerfile` (stage `mlflow-server`) |
| Config | `src/config/settings.py`, `.env.example`, `scripts/validate_env.py` |
| Serving | `src/serving/{api,model_loader,store,recommender}.py` |
| Testes | `tests/test_{preprocess,feature_engineering,labeling,ranking,ncf,contract,smoke,registry,serving}.py` |
| Docker | `Dockerfile`, `docker-compose.yml` |
