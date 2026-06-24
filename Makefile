.PHONY: env install lint format test setup api \
        dvc-repro dvc-pull dvc-push \
        mlflow compose-build compose-full compose-down \
        validate-env preprocess feature-eng train evaluate

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

# ─── DVC ──────────────────────────────────────────────────────────────────────

dvc-repro:
	poetry run dvc repro

dvc-pull:
	poetry run dvc pull

dvc-push:
	poetry run dvc push

# ─── Serviços Locais ──────────────────────────────────────────────────────────

mlflow:
	poetry run mlflow server \
		--host 0.0.0.0 \
		--port 5000 \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlartifacts

# Opção A — API local (usa código e modelo do host diretamente)
api:
	poetry run uvicorn src.serving.api:app \
		--host 0.0.0.0 --port 8000 --reload

# ─── Docker ───────────────────────────────────────────────────────────────────

# Opção B — Docker (requer rebuild para incorporar modelo e código atualizados)
compose-build:
	docker compose build

# Sobe MLflow + treino + API
compose-full:
	docker compose up

compose-down:
	docker compose down

# ─── Utilitários ──────────────────────────────────────────────────────────────

validate-env:
	poetry run python scripts/validate_env.py
