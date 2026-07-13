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

# Deve casar com a versão do cliente (poetry.lock) — o cliente 3.x resolve
# URIs runs:/ via a API "logged-models", ausente em servidores 2.x.
RUN pip install --no-cache-dir mlflow==3.13.0

EXPOSE 5000

# --allowed-hosts: MLflow 3.x valida o header Host (proteção anti DNS-rebinding).
# Na rede do compose o cliente acessa via 'mlflow:5000'; liberamos todos os hosts
# por ser um ambiente local. Para deploy público, restrinja a hosts específicos.
# --serve-artifacts + --artifacts-destination: os clientes (containers de
# train/evaluate/promote) sobem artefatos PELO servidor (proxy), pois não
# compartilham o filesystem /mlartifacts. Sem isso, log_artifact grava num
# caminho local inexistente no container cliente e nada chega ao servidor.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--allowed-hosts", "*", \
     "--backend-store-uri", "sqlite:////mlartifacts/mlflow.db", \
     "--serve-artifacts", \
     "--artifacts-destination", "/mlartifacts"]


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
