"""Task runner for Windows (alternative to Makefile)."""

import subprocess
import sys


def run(cmd: str) -> None:
    """Execute a shell command."""
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


tasks = {
    "env": "uv sync --extra dev",
    "install": "uv sync --extra dev",
    "lint": "uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/",
    "format": "uv run ruff check --fix src/ tests/ && uv run ruff format src/ tests/",
    "test": "uv run pytest tests/ -v",
    "test-cov": "uv run pytest tests/ --cov=src --cov-report=html",
    "dvc-repro": "dvc repro",
    "dvc-pull": "dvc pull",
    "mlflow": (
        "uv run mlflow server --host 0.0.0.0 --port 5000 "
        "--backend-store-uri sqlite:///mlflow.db "
        "--default-artifact-root ./mlartifacts"
    ),
    "docker-build": "docker compose build",
    "docker-up": "docker compose up -d",
    "docker-down": "docker compose down",
    "validate-env": "uv run python scripts/validate_env.py",
    "preprocess": "uv run python -m src.data.preprocess",
    "feature-eng": "uv run python -m src.features.build_features",
    "train": "uv run python -m src.training.trainer",
    "evaluate": "uv run python -m src.evaluation.evaluate",
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in tasks:
        print("Available tasks:", ", ".join(tasks.keys()))
        sys.exit(1)
    run(tasks[sys.argv[1]])
