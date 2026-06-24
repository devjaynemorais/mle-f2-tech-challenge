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
| Empacotamento | Poetry + pyproject.toml |
| Linting | ruff + pre-commit |
| Containerização | Docker multi-stage + docker-compose |
| Configuração | Pydantic Settings + params.yaml |

## Estrutura do Projeto

```
mle-f2-tech-challenge/
├── config/               # config.yaml (parâmetros estáticos)
├── data/
│   ├── raw/              # Dados brutos imutáveis (rastreados pelo DVC)
│   ├── interim/          # Dados intermediários limpos
│   └── processed/        # Splits train/val/test
├── docs/                 # model_card.md, eda.md, preprocessing.md, tests.md
├── metrics/              # Métricas DVC (JSON)
├── models/
│   └── artifacts/        # Artefatos do modelo serializado
├── notebooks/            # EDA e exploração
├── scripts/              # validate_env.py
├── src/
│   ├── config/           # settings.py (Pydantic Settings + .env)
│   ├── data/             # preprocess.py, feature_engineering.py, dataset.py
│   ├── features/         # build_features.py (Etapa 2)
│   ├── models/           # base.py (Factory), mlp.py, baselines.py
│   ├── training/         # trainer.py (Etapa 3)
│   ├── evaluation/       # evaluate.py (Etapa 4)
│   └── utils/            # logging, mlflow_tracking, seed
├── tests/
├── dvc.yaml              # Definição do pipeline DVC
├── params.yaml           # Parâmetros dos experimentos
├── pyproject.toml
├── Dockerfile            # Multi-stage
└── docker-compose.yml
```

## Início Rápido

**Pré-requisitos:** Python 3.11, Make, Docker

```bash
# 1. Criar ambiente virtual e instalar dependências
make env

# 2. Copiar e configurar variáveis de ambiente
cp .env.example .env

# 3. Obter os dados (DVC remote ou Kaggle API)
dvc pull
# ou: python scripts/download_dataset.py

# 4. Rodar pipeline completo (validate-env + dvc repro)
make setup

# 5. Iniciar a UI do MLflow
make mlflow
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

## Servir a API

**Opção A — local** (usa código e modelo do host diretamente):

```bash
make api
```

**Opção B — Docker** (requer rebuild para incorporar modelo e código atualizados):

```bash
make compose-build   # builda as imagens
make compose-full    # sobe MLflow + treino + API
make compose-down    # para todos os serviços
```

## Desenvolvimento

```bash
make lint        # ruff check + verificação de formatação
make format      # corrigir problemas de lint automaticamente
make test        # pytest
make test-cov    # pytest com relatório de cobertura em HTML
```

## Dataset

**RetailRocket E-commerce Dataset** — interações de usuários em loja virtual (visualizações, adições ao carrinho, compras).

| Arquivo | Descrição |
|---------|-----------|
| `events.csv` | Interações usuário-item com timestamp e tipo de evento |
| `item_properties_part1/2.csv` | Propriedades dos itens ao longo do tempo |
| `category_tree.csv` | Hierarquia de categorias |

Download: [kaggle.com/datasets/retailrocket/ecommerce-dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

Após baixar, colocar os arquivos em `data/raw/` e rodar `dvc repro`.

## Design Patterns

- **Factory** (`src/models/factory.py`): `ModelFactory.create("mlp")` instancia qualquer recomendador registrado via decorator `@ModelFactory.register`.
- **Strategy** (`src/data/preprocessor.py`): subclasses de `PreprocessStrategy` (ex: `RetailRocketPreprocessor`) trocam a lógica de pré-processamento sem alterar o pipeline.

## Critérios de Avaliação

| Critério | Peso |
|----------|------|
| Clean code e estrutura | 15% |
| Reprodutibilidade (Poetry, lock file, .env) | 15% |
| Docker (multi-stage, compose) | 15% |
| DVC + Pipeline (≥ 3 etapas, dvc repro) | 15% |
| Rede neural (PyTorch MLP) | 15% |
| MLflow + Registry | 10% |
| Vídeo STAR (5 min) | 10% |
| Bônus: deploy em nuvem | 5% |
