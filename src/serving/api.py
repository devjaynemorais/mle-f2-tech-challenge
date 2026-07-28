"""API FastAPI de serving do sistema de recomendação.

Endpoints:
  - GET /            → metadados do serviço
  - GET /health      → estado de carregamento do modelo
  - GET /recommend   → recomendações top-K para um usuário
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.config.settings import settings
from src.serving.model_loader import load_production_model
from src.serving.recommender import RecommendationService
from src.serving.store import FeatureStore

logger = logging.getLogger(__name__)

state: dict[str, RecommendationService | None] = {"service": None}

_DEMO_HTML_PATH = Path(__file__).parent / "static" / "demo.html"


class Recommendation(BaseModel):
    """Um item recomendado com seu score."""

    item_idx: int
    score: float


class RecommendResponse(BaseModel):
    """Resposta do endpoint /recommend."""

    user_id: int
    strategy: str
    count: int
    recommendations: list[Recommendation]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carrega modelo e feature store uma única vez no startup."""
    state["service"] = None
    try:
        model = load_production_model()
        store = FeatureStore.from_processed(Path(settings.data_processed_path))
        state["service"] = RecommendationService(
            model, store, settings.serving_max_candidates
        )
        logger.info(
            "Serviço pronto: %d usuários, %d itens", store.n_users, store.n_items
        )
    except Exception:  # noqa: BLE001 — API sobe em modo degradado
        logger.exception("Falha ao carregar o serviço; /health reportará degradado.")
    yield


app = FastAPI(title="RetailRocket Recommender API", version="0.1.0", lifespan=lifespan)


def _get_service() -> RecommendationService:
    """Retorna o serviço ou 503 se ainda não carregado."""
    service = state["service"]
    if service is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    return service


@app.get("/")
def root() -> dict:
    """Metadados do serviço."""
    return {"service": "RetailRocket Recommender API", "version": "0.1.0"}


@app.get("/health")
def health() -> dict:
    """Estado de carregamento do modelo e tamanho do catálogo."""
    service = state["service"]
    if service is None:
        return {"status": "degraded", "model_loaded": False}
    return {
        "status": "ok",
        "model_loaded": True,
        "n_users": service.store.n_users,
        "n_items": service.store.n_items,
    }


@app.get("/recommend", response_model=RecommendResponse)
def recommend(user_id: int, top_k: int = Query(10, ge=1, le=100)) -> dict:
    """Retorna as top-K recomendações para o usuário."""
    return _get_service().recommend(user_id, top_k)


@app.get("/explain")
def explain(user_id: int, top_k: int = Query(10, ge=1, le=100)) -> dict:
    """Retorna o passo a passo de como a recomendação foi montada.

    Usado pela demo interativa (``/demo``) — expõe as features cruas por
    candidato pontuado, o tamanho do pool e por que caiu no fallback de
    popularidade quando é o caso. Não é consumido pelo cliente de produto.
    """
    return _get_service().explain(user_id, top_k)


@app.get("/demo/sample-users")
def sample_users() -> dict:
    """Retorna usuários reais de exemplo (ativo, esparso, desconhecido)."""
    return _get_service().store.sample_users()


@app.get("/demo/user-timeline")
def user_timeline(user_id: int, limit: int = Query(15, ge=1, le=100)) -> dict:
    """Retorna o histórico real (últimos ``limit`` eventos) de um usuário.

    Cada evento vem com o perfil do item (categoria, popularidade) — dá
    visibilidade ao "quem é esse usuário" antes de pedir a recomendação.
    """
    store = _get_service().store
    return {
        "user_id": user_id,
        "known": store.has_user(user_id),
        "events": store.timeline(user_id, limit),
    }


@app.get("/demo", response_class=HTMLResponse)
def demo() -> str:
    """Serve a página HTML da demo interativa (self-contained, sem build)."""
    return _DEMO_HTML_PATH.read_text(encoding="utf-8")
