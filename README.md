# mle-f2-tech-challenge

Sistema de recomendação de produtos para e-commerce — FIAP MLE Fase 2.

## Problema

Uma empresa de e-commerce precisa de um sistema de recomendação de produtos baseado no comportamento de navegação dos usuários. O modelo central é uma rede neural (MLP ou embedding-based) treinada com PyTorch, com pipeline completo containerizado em Docker, dados versionados com DVC e experimentos rastreados no MLflow.

## Stack

| Camada | Ferramenta |
|--------|------------|
| Modelo | PyTorch (MLP), Scikit-Learn (baselines) |
| Rastreamento de experimentos | MLflow |
| Versionamento de dados | DVC |
| Empacotamento | uv + pyproject.toml |
| Linting | ruff + pre-commit |
| Containerização | Docker multi-stage + docker-compose |
| Configuração | Pydantic Settings + params.yaml |

## Estrutura do Projeto

```
mle-f2-tech-challenge/
├── config/               # Pydantic Settings + config.yaml
├── data/
│   ├── raw/              # Dados brutos imutáveis (rastreados pelo DVC)
│   ├── interim/          # Dados intermediários limpos
│   └── processed/        # Splits train/val/test
├── docs/                 # model_card.md + PDF dos requisitos
├── metrics/              # Métricas DVC (JSON)
├── models/
│   └── artifacts/        # Artefatos do modelo serializado
├── notebooks/            # EDA e exploração
├── scripts/              # validate_env.py
├── src/
│   ├── data/             # preprocess.py (Etapa 1)
│   ├── features/         # build_features.py (Etapa 2)
│   ├── models/           # base.py (Factory), mlp.py, baselines.py
│   ├── training/         # trainer.py (Etapa 3)
│   ├── evaluation/       # evaluate.py (Etapa 4)
│   └── utils/            # logging, helpers do MLflow
├── tests/
├── dvc.yaml              # Definição do pipeline DVC
├── params.yaml           # Parâmetros dos experimentos
├── pyproject.toml
├── Dockerfile            # Multi-stage
└── docker-compose.yml
```

## Início Rápido

```bash
# 1. Instalar dependências
uv sync --extra dev

# 2. Copiar e configurar o ambiente
cp .env.example .env

# 3. Validar o ambiente
make validate-env

# 4. Colocar o dataset em data/raw/ e rodar o pipeline completo
make dvc-repro

# 5. Iniciar a UI do MLflow
make mlflow
```

### Windows (sem make)

```powershell
python tasks.py env
python tasks.py validate-env
python tasks.py dvc-repro
python tasks.py mlflow
```

## Pipeline DVC

```
preprocess → feature_eng → train → evaluate
```

Rodar cada etapa individualmente:

```bash
make preprocess
make feature-eng
make train
make evaluate
```

Ou reproduzir o pipeline completo:

```bash
dvc repro
```

## Docker

```bash
# Build e iniciar MLflow + serviço de treino
make docker-up

# Parar todos os serviços
make docker-down
```

## Desenvolvimento

```bash
make lint       # ruff check + verificação de formatação
make format     # corrigir problemas de lint automaticamente
make test       # pytest
make test-cov   # pytest com relatório de cobertura em HTML
```

## Dataset

> A definir — ainda não selecionado. Candidatos:
> - [Instacart Market Basket](https://www.kaggle.com/c/instacart-market-basket-analysis)
> - [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
> - [MovieLens](https://grouplens.org/datasets/movielens/)
>
> Qualquer dataset com ≥ 10.000 interações usuário-item é aceito.

## Design Patterns

- **Factory** (`src/models/base.py`): `ModelFactory.create("mlp")` instancia qualquer recomendador registrado.
- **Strategy** (`src/data/preprocess.py`): subclasses de `PreprocessStrategy` trocam a lógica de pré-processamento sem alterar o pipeline.

## Critérios de Avaliação

| Critério | Peso |
|----------|------|
| Clean code e estrutura | 15% |
| Reprodutibilidade (uv, lock file, .env) | 15% |
| Docker (multi-stage, compose) | 15% |
| DVC + Pipeline (≥ 3 etapas, dvc repro) | 15% |
| Rede neural (PyTorch MLP) | 15% |
| MLflow + Registry | 10% |
| Vídeo STAR (5 min) | 10% |
| Bônus: deploy em nuvem | 5% |
