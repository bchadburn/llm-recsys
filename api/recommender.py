"""Singleton recommender loaded once at startup and reused across requests."""
from __future__ import annotations

import logging

import numpy as np

from data import (
    PRICE_SENS_IDX,
    PRICE_TIERS,
    InteractionDataset,
    generate_data,
    get_item_text_embeddings,
)
from main import build_faiss_index, train
from ranker import RETRIEVAL_K, _embed_items, _embed_users, _make_features, train_ranker

logger = logging.getLogger(__name__)


class Recommender:
    """Encapsulates the two-tower + LightGBM ranker inference path."""

    def __init__(self) -> None:
        logger.info("Generating synthetic data and training models...")
        user_features, item_features, items, interactions, _ = generate_data(
            n_users=500, n_items=200, n_interactions=5000
        )
        item_text_embs = get_item_text_embeddings(items)

        self._user_features = user_features
        self._item_features = item_text_embs
        self._items = items
        self._interactions = interactions
        self._n_users = len(user_features)

        logger.info("Training two-tower model...")
        self._user_tower, self._item_tower = train(
            user_features, item_text_embs, interactions, InteractionDataset
        )
        logger.info("Building FAISS index...")
        self._faiss_index = build_faiss_index(self._item_tower, item_text_embs)

        logger.info("Training LightGBM ranker...")
        self._ranker = train_ranker(
            self._user_tower,
            self._item_tower,
            self._faiss_index,
            user_features,
            item_text_embs,
            items,
            interactions,
            price_sens_idx=PRICE_SENS_IDX,
        )

        self._user_embs = _embed_users(self._user_tower, user_features)
        self._item_embs = _embed_items(self._item_tower, item_text_embs)
        logger.info("Recommender ready. %d users, %d items.", self._n_users, len(items))

    def recommend(self, user_id: int, top_k: int) -> list[dict]:
        if user_id < 0 or user_id >= self._n_users:
            raise ValueError(f"user_id {user_id} out of range [0, {self._n_users - 1}]")

        query_emb = self._user_embs[user_id : user_id + 1]
        faiss_scores, faiss_indices = self._faiss_index.search(query_emb, RETRIEVAL_K)
        faiss_scores = faiss_scores[0]
        faiss_indices = faiss_indices[0]

        price_sens = float(self._user_features[user_id][PRICE_SENS_IDX])
        X = _make_features(
            self._user_embs[user_id],
            self._item_embs,
            faiss_scores,
            faiss_indices,
            price_sens,
            self._items,
        )
        lgb_scores = self._ranker.predict(X)

        order = np.argsort(-lgb_scores)[: top_k]
        results = []
        for rank_idx in order:
            item_idx = int(faiss_indices[rank_idx])
            item = self._items[item_idx]
            price_tier_name = PRICE_TIERS[int(item["price_tier"])]
            results.append(
                {
                    "item_id": item_idx,
                    "name": item["name"],
                    "category": item["category"],
                    "price_tier": price_tier_name,
                    "score": float(lgb_scores[rank_idx]),
                }
            )
        return results
