.PHONY: env install lint format test dvc-repro dvc-pull mlflow docker-build docker-up docker-down validate-env

env:
	uv sync --extra dev

install:
	uv sync --extra dev

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ --cov=src --cov-report=html

dvc-init:
	dvc init

dvc-repro:
	dvc repro

dvc-pull:
	dvc pull

dvc-push:
	dvc push

mlflow:
	uv run mlflow server \
		--host 0.0.0.0 \
		--port 5000 \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlartifacts

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

validate-env:
	uv run python scripts/validate_env.py

preprocess:
	uv run python -m src.data.preprocess

feature-eng:
	uv run python -m src.features.build_features

train:
	uv run python -m src.training.trainer

evaluate:
	uv run python -m src.evaluation.evaluate
