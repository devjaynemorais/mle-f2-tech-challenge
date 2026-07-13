.PHONY: env install lint format test setup api \
        dvc-repro dvc-pull dvc-push \
        mlflow compose-build compose-full compose-down \
        compose-pipeline compose-promote \
        validate-env preprocess feature-eng train evaluate promote

# ─── Ambiente ─────────────────────────────────────────────────────────────────

env:
	pip install poetry==1.8.3 --quiet
	poetry install --with dev

install: env

# ─── Qualidade de Código ──────────────────────────────────────────────────────

lint:
	poetry run ruff check src/ tests/
	poetry run ruff format --check src/ tests/

format:
	poetry run ruff check --fix src/ tests/
	poetry run ruff format src/ tests/

# ─── Testes ───────────────────────────────────────────────────────────────────

test:
	poetry run pytest tests/ -v

test-cov:
	poetry run pytest tests/ --cov=src --cov-report=html

# ─── Pipeline Completo ────────────────────────────────────────────────────────

# Equivalente ao 'make setup' do outro projeto:
# executa todo o pipeline de dados → features → treino → avaliação via DVC
setup: validate-env dvc-repro

# ─── Estágios individuais do pipeline ─────────────────────────────────────────

preprocess:
	poetry run python -m src.data.preprocess

feature-eng:
	poetry run python -m src.features.build_features

train:
	poetry run python -m src.training.trainer

evaluate:
	poetry run python -m src.evaluation.evaluate

# Registra o melhor run no MLflow Registry e promove Staging → Production
promote:
	poetry run python -m src.models.registry

# ─── DVC ──────────────────────────────────────────────────────────────────────

dvc-repro:
	poetry run dvc repro

dvc-pull:
	poetry run dvc pull

dvc-push:
	poetry run dvc push

# ─── Serviços Locais ──────────────────────────────────────────────────────────

# Porta parametrizável (padrão 5000). Se a 5000 estiver ocupada, rode:
#   make mlflow MLFLOW_PORT=5001   (e ajuste MLFLOW_TRACKING_URI no .env)
MLFLOW_PORT ?= 5000
mlflow:
	poetry run mlflow server \
		--host 0.0.0.0 \
		--port $(MLFLOW_PORT) \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlartifacts

# Opção A — API local (usa código e modelo do host diretamente)
# Porta parametrizável (padrão 8000). Se a 8000 estiver ocupada (ex.: Docker
# Desktop), rode:  make api API_PORT=8001
API_PORT ?= 8000
api:
	poetry run uvicorn src.serving.api:app \
		--host 0.0.0.0 --port $(API_PORT) --reload

# ─── Docker ───────────────────────────────────────────────────────────────────

# Opção B — Docker (requer rebuild para incorporar modelo e código atualizados)
compose-build:
	docker compose build

# Sobe MLflow + treino + API
compose-full:
	docker compose up

# Pipeline de ML no Docker: sobe o MLflow e roda train → evaluate → promote
# como jobs one-shot (não sobe a API). Requer .env (copie de .env.example).
compose-pipeline:
	docker compose up -d mlflow
	docker compose run --rm train
	docker compose run --rm evaluate
	docker compose run --rm promote
	@echo "Modelo promovido — veja models/promoted_model.json e a aba Models em http://localhost:5000"

# Apenas o passo de promoção no Docker (assume que já houve treino no MLflow)
compose-promote:
	docker compose up -d mlflow
	docker compose run --rm promote

compose-down:
	docker compose down

# ─── Utilitários ──────────────────────────────────────────────────────────────

validate-env:
	poetry run python scripts/validate_env.py
