# Tech Challenge — Fase 02
> Sistema de Recomendação de Produtos · Retailrocket Dataset

> **Última revisão do todo:** 2026-07-05 (branch `feat/neural-model-registry`)
>
> **Estado geral:** pipeline DVC de 5 stages (`preprocess → feature_eng → train → evaluate → promote`)
> funcional; MLflow tracking + Model Registry (Staging → Production) implementados e testados.
> **Principais lacunas:** (1) API de serving `src/serving/api.py` **não existe** — Dockerfile stage
> `api` e serviço compose `api` apontam para um módulo ausente; (2) baselines em `src/models/` são
> apenas `dummy` + `logistic` (não Popularity/SVD/KNN); (3) métricas em `evaluate.py` são de
> classificação binária (ROC-AUC, AP, F1, precision, recall), **não** as de ranking Top-K
> (Precision@K/NDCG@K/MRR); (4) MLP não usa `nn.Embedding` (features numéricas cruas);
> (5) `docs/model_card.md` é um template não preenchido.

---

## Etapa 1 — Clean Code + Ambiente

### Repositório e estrutura
- [x] Criar repositório GitHub e configurar branch protection na `main`
- [x] Criar estrutura de pastas: `src/`, `tests/`, `data/`, `models/`, `configs/`, `scripts/`, `notebooks/`
- [x] Definir padrão de commits semânticos (`feat:`, `fix:`, `chore:`, `docs:`, `test:`)

### Dependências e ambiente
- [x] Configurar `pyproject.toml` com Poetry (deps prod + dev separadas)
- [x] Gerar e commitar `poetry.lock`
- [x] Criar `.env.example`, `.gitignore`, `.dockerignore`
- [x] Configurar `src/config/settings.py` com Pydantic Settings + leitura do `.env`
- [x] Criar `scripts/validate_env.py` (valida variáveis de ambiente e dependências)
- [x] Fixar seeds: `torch.manual_seed`, `numpy.random.seed`, `random.seed` em `src/utils/seed.py`

### Qualidade de código
- [x] Configurar `ruff` no `pyproject.toml`
- [x] Configurar `pre-commit` hooks (ruff, mypy)
- [x] Garantir funções ≤ 20 linhas em todo o projeto
- [x] Type hints em todas as funções públicas (presentes de forma consistente no código revisado)
- [x] Docstrings padrão Google style em todas as funções públicas (idem)
- [x] `ruff check src/` passando sem erros

### Design patterns (obrigatório)
- [x] Implementar `ModelFactory` em `src/models/factory.py` (cria MLP ou EmbeddingModel via `params.yaml`)
- [x] Implementar `PreprocessorStrategy` em `src/data/preprocessor.py` (diferentes estratégias de feature engineering)

---

## Etapa 2 — Pipeline de Dados + DVC

### Exploração e dados
- [X] Download do Retailrocket dataset (events.csv, item_properties, category_tree)
- [X] Exploração inicial em `notebooks/01_exploratory_data_analysis.ipynb`
- [x] Documentar schema e características do dataset no README (seção "Dataset" + `docs/eda.md`)
- [X] `dvc init` no repositório
- [X] `dvc add data/raw/` (versionar dados brutos)
- [X] Configurar DVC remote local: `dvc remote add -d local_remote dvc-storage`

### Pré-processamento
- [x] Implementar `src/data/preprocess.py` (funções ≤ 20 linhas, type hints)
  - [x] Filtrar usuários com menos de N interações
  - [x] Encoding de `user_id` e `item_id` para índices
  - [x] Salvar artefatos em `data/interim/`
- [x] Implementar `src/data/feature_engineering.py`
  - [x] Gerar features de comportamento (recência, frequência, tipo de evento)
  - [x] Salvar features em `data/processed/` (via `build_features.py`)
- [x] Implementar `src/data/dataset.py` (PyTorch Dataset + DataLoader)
- [x] Implementar split cronológico treino/validação/teste

### Pipeline DVC
- [x] Criar `dvc.yaml` com os stages e `deps`/`outs` corretos (5 stages — `promote` adicionado):
  - [x] `preprocess`
  - [x] `feature_eng`
  - [x] `train`
  - [x] `evaluate`
  - [x] `promote` (registra + promove modelo no MLflow Registry)
- [x] Criar `params.yaml` com hiperparâmetros versionados (inclui seção `registry`)
- [x] Garantir que `dvc repro` roda do zero sem erro (`dvc.lock` + outputs presentes; obs.: stage `promote` exige o servidor MLflow acessível em `MLFLOW_TRACKING_URI`)

### Testes
- [x] Escrever `tests/test_preprocess.py` (11)
- [x] Escrever `tests/test_feature_engineering.py` (23)
- [x] Escrever `tests/test_smoke.py` (4 — ModelFactory + MLP fit/predict)
- [x] Escrever `tests/test_registry.py` (3 — registro/promoção MLflow)
- [x] `pytest tests/ -v` passando (**41 testes** — README ainda cita 38, desatualizado)

---

## Etapa 3 — Docker + MLflow

### Docker
- [x] `Dockerfile` multi-stage: stage `builder` (poetry install) + stage `runtime` (app)
- [x] `docker-compose.yml` com:
  - [x] Serviço `train`
  - [x] Serviço `mlflow` (porta 5000)
  - [x] Serviço `api` (porta 8000)
  - [x] Volumes para persistência de artefatos
- [x] Validar `docker build` sem erros (mlflow, train, evaluate, promote)
- [~] Validar `docker-compose up` sobe todos os serviços (mlflow/train/evaluate/promote OK; serviço `api` **quebra** — Dockerfile stage `api` roda `uvicorn src.serving.api:app`, mas `src/serving/api.py` não existe)

### MLflow tracking
- [x] Integrar `mlflow.log_param()`, `mlflow.log_metric()`, `mlflow.log_artifact()` no treino
- [~] Rodar ≥ 3 experimentos com params distintos (≥ 4 runs já existem em `mlruns/1/`, mas ainda **não** de forma sistemática variando os params abaixo):
  - [ ] Run 1: `lr=0.001`, `embedding_dim=32`, `batch_size=256`
  - [ ] Run 2: `lr=0.0005`, `embedding_dim=64`, `batch_size=512`
  - [ ] Run 3: `lr=0.001`, `embedding_dim=128`, `batch_size=256`
  - Obs.: `embedding_dim` não é parâmetro atual do MLP (sem `nn.Embedding`); ajustar os params variados ao modelo real
- [x] MLflow UI acessível em `http://localhost:5000` (stage `mlflow-server` no Dockerfile + serviço compose com healthcheck)

---

## Etapa 4 — Modelo Neural + Registry + Entrega

### Baselines (Scikit-Learn)
- [~] Implementar `src/models/baselines.py` — **atual: `dummy` + `logistic`** (registrados na Factory); os baselines abaixo foram explorados só em `notebooks/02_exploration_MLP_Architecture.ipynb`, não no `src/`:
  - [ ] Popularity-based (existe no notebook, não no `src/`)
  - [ ] SVD (`TruncatedSVD`) — ou ALS/Item-CF do notebook, não portados para `src/`
  - [ ] KNN (`NearestNeighbors`)
- [~] Métricas de avaliação — `src/evaluation/evaluate.py` calcula **5 métricas de classificação binária** (ROC-AUC, Average Precision, F1, Precisão, Recall), **não** as de ranking Top-K que o spec pede. Falta `src/evaluation/metrics.py` com:
  - [ ] `Precision@K`
  - [ ] `Recall@K`
  - [ ] `NDCG@K`
  - [ ] `MRR`

### Modelo PyTorch
- [ ] Implementar MLP em `src/models/mlp.py`:
  - [ ] Camadas de embedding para user e item (atual: `user_idx`/`item_idx` entram como features numéricas cruas, sem `nn.Embedding`)
  - [x] Camadas densas com ReLU
  - [x] Camada de saída (score de relevância)
- [~] Implementar loop de treino em `src/training/trainer.py` (funcional; loga `val_auc` no MLflow):
  - [x] Early stopping com paciência configurável (em `MLPRecommender._run_early_stopping`)
  - [ ] Log de loss por época no MLflow (hoje só `val_auc` final é logado)
- [ ] Comparar MLP vs baselines com as 4 métricas de ranking (pipeline treina só 1 modelo por vez via `params.yaml`; comparação não automatizada)

### MLflow Registry
- [x] Registrar melhor modelo no MLflow Registry (`src/models/registry.py` + stage DVC `promote`; helpers em `src/utils/mlflow_tracking.py`; testado em `tests/test_registry.py`)
- [x] Promover modelo: **Staging → Production** (`register_model` → Staging, `promote_model` → Production; registro persistido em `models/promoted_model.json`)
- [x] Validar que a API carrega o modelo direto do Registry (Production) — `load_production_model()` Registry-first com fallback local

### API de serving
- [x] Implementar `src/serving/api.py` com FastAPI (+ `store.py`, `model_loader.py`, `recommender.py`)
  - [x] Endpoint `GET /recommend?user_id=X&top_k=10`
  - [x] Retorna JSON com lista de itens rankeados (+ `/health`, `/`)
- [x] Testar: `curl "http://localhost:8000/recommend?user_id=1&top_k=10"` (cobertura em `tests/test_serving.py`)

### Documentação
- [~] Escrever `MODEL_CARD.md` — existe em `docs/model_card.md` mas é **template não preenchido** (campos "A definir" / métricas "—"); falta preencher performance, dataset e versão reais
- [x] Finalizar `README.md` com instruções completas de execução (falta apenas o link do vídeo STAR)

### Vídeo STAR (≤ 5 minutos)
- [ ] Escrever roteiro:
  - [ ] **Situation**: problema de negócio + contexto do dataset
  - [ ] **Task**: objetivos técnicos e restrições
  - [ ] **Action**: decisões de arquitetura, modelo, versionamento e containerização
  - [ ] **Result**: métricas obtidas, trade-offs e lições aprendidas
- [ ] Gravar e fazer upload
- [ ] Adicionar link do vídeo no README

---

## Bônus — Deploy AWS Academy (+5%)
> ⚠️ Só iniciar se todas as etapas anteriores estiverem funcionando localmente.

- [ ] Criar instância EC2 `t3.medium` com Amazon Linux 2
- [ ] Instalar Docker e Docker Compose na EC2
- [ ] Configurar S3 bucket como DVC remote
  - [ ] Atualizar `.dvc/config` com remote S3
  - [ ] Configurar AWS credentials via `.env`
- [ ] Clonar repositório na EC2 e rodar `docker-compose up -d`
- [ ] Configurar Security Group:
  - [ ] Porta 22 (SSH)
  - [ ] Porta 8000 (FastAPI)
  - [ ] Porta 5000 (MLflow UI)
- [ ] Testar URL pública: `curl http://<EC2-IP>:8000/recommend?user_id=1&top_k=5`
- [ ] Adicionar URL pública no README

---

## Checklist de entrega final

- [ ] `poetry install` do zero funciona sem erros
- [ ] `dvc repro` reproduz o pipeline completo
- [ ] `docker-compose up` sobe todos os serviços
- [ ] ≥ 3 MLflow runs com params + métricas logados
- [ ] Modelo promovido a Production no MLflow Registry
- [ ] `ruff check src/` sem erros
- [ ] Todas as funções públicas com type hints e docstrings
- [ ] `.env.example` atualizado com todas as variáveis
- [ ] `MODEL_CARD.md` preenchido
- [ ] `README.md` com instruções completas
- [ ] Histórico de commits semântico no GitHub
- [ ] Link do vídeo STAR no README
- [ ] Submeter link do repositório

---

## Referência rápida

### Critérios e pesos
| Critério | Peso | Status |
|---|---|---|
| Clean code e estrutura | 20% | ⬜ |
| Reprodutibilidade (Poetry + lock + .env) | 15% | ⬜ |
| Docker multi-stage + compose | 15% | ⬜ |
| DVC + pipeline ≥ 4 stages | 15% | ⬜ |
| Rede neural PyTorch + early stopping | 15% | ⬜ |
| MLflow + Registry (Staging → Production) | 10% | ⬜ |
| Vídeo STAR ≤ 5 min | 10% | ⬜ |
| Bônus: deploy AWS com URL pública | +5% | ⬜ |

### Comandos do dia a dia
```bash
# Instalar dependências
poetry install

# Rodar pipeline completo
dvc repro

# Subir ambiente local
docker-compose up --build

# Verificar linting
ruff check src/

# Rodar testes
pytest tests/ -v

# Acessar MLflow UI
open http://localhost:5000

# Testar API
curl "http://localhost:8000/recommend?user_id=12345&top_k=10"
```

### Design patterns obrigatórios
- **Factory** → `src/models/factory.py` — cria MLP ou EmbeddingModel via `params.yaml`
- **Strategy** → `src/data/preprocessor.py` — diferentes estratégias de feature engineering

### DVC stages (dvc.yaml)
```
preprocess → feature_eng → train → evaluate → promote
```
