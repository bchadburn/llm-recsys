"""Pydantic request/response schemas for the recommendation API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=0, description="User ID (0-indexed)")
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations to return")


class RecommendedItem(BaseModel):
    item_id: int
    name: str
    category: str
    price_tier: str  # "budget" | "mid" | "premium"
    score: float = Field(..., description="Ranking score (higher = more relevant)")


class RecommendResponse(BaseModel):
    user_id: int
    top_k: int
    items: list[RecommendedItem]
    model: str = "two-tower+lightgbm"
