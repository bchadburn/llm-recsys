"""FastAPI inference service for the two-tower grocery recommendation system.

Startup trains models on synthetic data once; subsequent requests use the
cached models. For production use, replace with model checkpoint loading.

Usage:
    uvicorn api.main:app --reload          # dev
    docker compose up                      # production
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.models import RecommendRequest, RecommendResponse, RecommendedItem
from api.recommender import Recommender

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_recommender: Recommender | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _recommender
    logger.info("Loading recommendation models (this takes ~30s on first run)...")
    _recommender = Recommender()
    yield
    _recommender = None


app = FastAPI(
    title="Grocery Recommender API",
    description=(
        "Two-tower retrieval (FAISS) + LightGBM re-ranking for grocery recommendations. "
        "Trained on synthetic data by default; swap in Instacart data for production."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _recommender is not None}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    if _recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        items = _recommender.recommend(request.user_id, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return RecommendResponse(
        user_id=request.user_id,
        top_k=request.top_k,
        items=[RecommendedItem(**item) for item in items],
    )
