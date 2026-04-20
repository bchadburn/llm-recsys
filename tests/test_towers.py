"""Tests for UserTower and ItemTower embedding behavior."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import ItemTower, UserTower


def test_user_tower_output_is_l2_normalized():
    """Embeddings must be unit vectors (FAISS dot product = cosine similarity)."""
    tower = UserTower(input_dim=10, embed_dim=64)
    x = torch.randn(8, 10)
    emb = tower(x)
    norms = emb.norm(dim=-1)
    assert emb.shape == (8, 64)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_item_tower_output_is_l2_normalized():
    """Item embeddings must be unit vectors."""
    tower = ItemTower(input_dim=12, embed_dim=64)
    x = torch.randn(4, 12)
    emb = tower(x)
    norms = emb.norm(dim=-1)
    assert emb.shape == (4, 64)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_dot_product_similarity_in_range():
    """Cosine similarity of L2-normalized vectors is in [-1, 1]."""
    user_tower = UserTower(input_dim=10, embed_dim=32)
    item_tower = ItemTower(input_dim=10, embed_dim=32)
    users = user_tower(torch.randn(5, 10))
    items = item_tower(torch.randn(5, 10))
    scores = (users * items).sum(dim=-1)
    assert (scores >= -1.0 - 1e-5).all()
    assert (scores <= 1.0 + 1e-5).all()
