# Stage 1: builder — install production dependencies with Poetry
FROM python:3.11-slim AS builder

WORKDIR /app

ENV POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock* ./

RUN poetry install --only main --no-root && rm -rf "${POETRY_CACHE_DIR}"


# Stage 2: runtime — training and evaluation jobs
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY config/ config/
COPY params.yaml .

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "-m"]
CMD ["src.training.trainer"]


# Stage 3: mlflow-server — experiment tracking server
FROM python:3.11-slim AS mlflow-server

RUN pip install --no-cache-dir mlflow==2.12.0

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlartifacts/mlflow.db", \
     "--default-artifact-root", "/mlartifacts"]


# Stage 4: api — FastAPI serving endpoint
FROM python:3.11-slim AS api

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY config/ config/
COPY params.yaml .

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
