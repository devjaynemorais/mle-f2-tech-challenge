# Como rodar o projeto — guia completo

Passo a passo do zero até: pipeline reproduzido, gates validados, API
servindo recomendações. Complementa o [`README.md`](../README.md) (visão
geral) e o [`quickstart.md`](../specs/001-recommender-quality/quickstart.md)
da spec (validação dos critérios de aceite).

---

## 0. Pré-requisitos

| Requisito | Versão | Verificar |
|-----------|--------|-----------|
| Python | ≥ 3.11 | `python --version` |
| Poetry | 1.8.3 | `poetry --version` |
| Git | qualquer | `git --version` |
| Docker (opcional) | com compose v2 | `docker compose version` |

## 1. Instalar o ambiente

```bash
make env            # instala poetry==1.8.3 e roda poetry install --with dev
# ou, sem make:
pip install poetry==1.8.3
poetry install --with dev
```

Copie a configuração de ambiente e valide:

```bash
cp .env.example .env      # Windows: copy .env.example .env
make validate-env         # checa variáveis e dependências
```

O `.env` controla infraestrutura (URI do MLflow, caminhos, portas) via
Pydantic Settings — os hiperparâmetros ficam no `params.yaml`.

## 2. Obter o dataset

Baixe o [RetailRocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
e coloque em `data/raw/`:

```
data/raw/
├── events.csv                   (~90 MB)  ← obrigatório
├── item_properties_part1.csv    (~460 MB) ← usado pelo estágio content (categoria)
├── item_properties_part2.csv    (~390 MB) ← idem
└── category_tree.csv            (~14 KB)  (não usado pelo pipeline)
```

> Sem os `item_properties_*`, o estágio `content` gera um parquet vazio e o
> modelo treina sem embedding de categoria (com aviso no log). Sem o
> `events.csv` o pipeline não roda.

Se o time usa um remote DVC configurado: `dvc pull` substitui o download
manual.

## 3. Subir o MLflow (antes do pipeline)

Os estágios `train`, `evaluate` e `promote` logam no MLflow Tracking Server
apontado por `MLFLOW_TRACKING_URI` (default `http://localhost:5000`).

```bash
make mlflow     # server local com backend SQLite (mlflow.db) na porta 5000
```

Deixe rodando num terminal separado. UI em <http://localhost:5000>.

> Porta 5000 ocupada? `make mlflow MLFLOW_PORT=5001` e ajuste
> `MLFLOW_TRACKING_URI=http://localhost:5001` no `.env`.

## 4. Rodar o pipeline completo (DVC)

```bash
dvc repro       # ou: make dvc-repro / make setup (valida o env antes)
```

Estágios, em ordem (definidos em `dvc.yaml`):

| # | Estágio | O que faz | Saída |
|---|---------|-----------|-------|
| 1 | `preprocess` | Filtra usuários < 5 interações, encoda ids, ordena por tempo | `data/interim/data_clean.parquet` |
| 2 | `content` | Extrai `categoryid` mais recente por item dos `item_properties_*` (chunked) | `data/content/item_categories.parquet` |
| 3 | `feature_eng` | Features **causais** (*as-of*) + categoria + split cronológico 70/10/20 | `data/processed/{train,val,test}.parquet` |
| 4 | `train` | Labeling (positivos + negativos 8:1) → NCF com early stopping por val-AUC → loga `val_auc` e `val_ndcg_at_20` | `models/artifacts/<run_id>/model.pkl`, `metrics/train_metrics.json` |
| 5 | `evaluate` | Classificação + ranking Top-K por usuário (forte/ampla, warm/cold) vs. baseline de popularidade | `metrics/eval_metrics.json`, `metrics/plots/` |
| 6 | `promote` | Melhor run por `val_ndcg_at_20` → Registry Staging → Production | `models/promoted_model.json` |

O DVC só reexecuta o que mudou (código, dados ou `params.yaml`). Duração
aproximada na base completa: ~8–12 min (o `content` domina a primeira
execução; depois fica cacheado).

### ⚠️ Windows: dois cuidados

1. **Use o Python da venv no PATH** — o DVC invoca `python` filho; se o
   PATH resolver outro Python, os estágios falham com
   `ModuleNotFoundError`. No PowerShell:

   ```powershell
   $env:PATH = "$PWD\.venv\Scripts;$env:PATH"
   ```

   (Com `poetry run dvc repro` ou `make dvc-repro` isso já é automático.)

2. **Force UTF-8 no console** — o MLflow imprime emoji na URL do run e
   quebra no console cp1252 (`UnicodeEncodeError` no fim do treino):

   ```powershell
   $env:PYTHONUTF8 = "1"
   ```

Comando completo no PowerShell:

```powershell
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"; $env:PYTHONUTF8 = "1"; dvc repro
```

### Estágios individuais (debug)

```bash
make preprocess      # python -m src.data.preprocess
poetry run python -m src.data.content_etl
make feature-eng     # python -m src.features.build_features
make train           # python -m src.training.trainer
make evaluate        # python -m src.evaluation.evaluate
make promote         # python -m src.models.registry
```

## 5. Conferir métricas e critérios de aceite

```bash
dvc metrics show                 # ou abra metrics/eval_metrics.json
```

- **AC-2 (classificação)**: `classification.roc_auc ≥ 0.70` no teste.
- **AC-1 (ranking, cenário principal)**: em `ranking.strong.model.warm`,
  `ndcg_at_{10,20}` e `recall_at_{10,20}` maiores que os equivalentes em
  `ranking.strong.popularity.warm`.
- Segmentos: `warm` = usuário com histórico no treino; `cold` = sem
  (cold é coberto pelo fallback de popularidade no serving).
- Números de referência e limitações: [`model_card.md`](model_card.md).

No MLflow (<http://localhost:5000>): runs de treino (métricas
`val_auc`/`val_ndcg_at_20`), run `evaluate` (métricas achatadas) e a aba
**Models** com `retailrocket_recommender` em Production.

## 6. Qualidade: testes e lint

```bash
make test        # pytest tests/ -v  (79 testes, todos com dados sintéticos)
make test-cov    # cobertura HTML em htmlcov/
make lint        # ruff check + ruff format --check
make format      # aplica correções automaticamente
```

Nenhum teste lê `data/` nem exige MLflow rodando — passam em clone limpo.
O teste de ausência de vazamento temporal (AC-3):
`poetry run pytest tests/test_feature_engineering.py -k leakage`.

## 7. Servir a API

```bash
make api     # uvicorn src.serving.api:app na porta 8000, com hot-reload
```

No startup a API carrega o modelo Production do MLflow Registry; se o
servidor estiver fora do ar, cai para o artefato local apontado por
`models/promoted_model.json` (pré-check TCP de 2 s). O feature store em
memória é montado a partir de `data/processed/`.

| Endpoint | Descrição |
|----------|-----------|
| `GET /` | Metadados do serviço |
| `GET /health` | `{status, model_loaded, n_users, n_items}` |
| `GET /recommend?user_id=X&top_k=10` | Top-K itens (`top_k` ∈ [1, 100]) |

```bash
curl "http://localhost:8000/recommend?user_id=42&top_k=10"
```

- Usuário com histórico → `strategy: "model"` (candidatos = top-5000
  populares não vistos, pontuados pelo NCF).
- Usuário desconhecido ou candidatos esgotados → `strategy: "popularity"`
  (nunca lista vazia).

Coleção Postman pronta em `postman/`.

> Porta 8000 ocupada? `make api API_PORT=8001`.

## 8. Docker (alternativa ao fluxo local)

```bash
make compose-build      # builda a imagem compartilhada
make compose-pipeline   # sobe mlflow e roda train → evaluate → promote
make compose-full       # sobe mlflow + train + api
make compose-down       # derruba tudo
```

Requer `.env` (copie de `.env.example`; dentro do compose o tracking URI é
`http://mlflow:5000`). Os dados de `data/processed/` precisam existir no
host (o compose monta o diretório) — rode os estágios de dados antes ou use
`dvc pull`.

## 9. Mexer nos hiperparâmetros

Tudo em `params.yaml` (o DVC detecta a mudança e reexecuta só o necessário):

| Bloco | Chaves principais |
|-------|-------------------|
| `labeling` | `num_negatives` (8), `popularity_alpha` (0 = uniforme — **não** volte para 0.75 sem ler o model card), `seed` |
| `train` | `model_type` (ncf/logistic/dummy), `epochs`, `embedding_dim`, `hidden_dims`, `dropout`, `unknown_dropout`, `weight_decay`, `learning_rate`, `early_stopping_patience` |
| `eval` | `k_values` ([10, 20]), `num_candidate_negatives` (100), `max_broad_positives` (20), `seed` |
| `registry` | `metric` (`val_ndcg_at_20`), `stage` (Production) |

Depois: `dvc repro` → conferir `metrics/eval_metrics.json` → o `promote`
escolhe o **melhor run histórico** do experimento pela métrica do registry.

> Nota: se você quer que o modelo recém-treinado seja o promovido mesmo
> não sendo o melhor histórico, troque `mlflow.experiment_name` (começa um
> experimento limpo) ou apague os runs antigos na UI do MLflow.

## 10. Troubleshooting

| Sintoma | Causa | Correção |
|---------|-------|----------|
| `ModuleNotFoundError` num estágio DVC | `python` do PATH ≠ venv | `poetry run dvc repro` ou prefixar o PATH (seção 4) |
| `UnicodeEncodeError ... charmap` no fim do treino | console Windows cp1252 × emoji do MLflow | `PYTHONUTF8=1` |
| `train` falha com conexão recusada | MLflow não está de pé | `make mlflow` antes do `dvc repro` |
| API sobe "degradada" (`/health` → `degraded`) | sem modelo Production nem `promoted_model.json` | rode o pipeline até `promote` |
| `dvc repro` reexecuta tudo do nada | mudou `params.yaml`/código rastreado | esperado — confira `dvc status` |
| Porta 5000/8000 ocupada | outro serviço | `MLFLOW_PORT=`/`API_PORT=` (seções 3 e 7) |
