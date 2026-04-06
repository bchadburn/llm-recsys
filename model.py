import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """
    Encodes a user feature vector into a 64-dim L2-normalized embedding.

    When n_users is provided, a learned nn.Embedding(n_users, id_embed_dim) table
    is concatenated with the hand-crafted feature vector before the MLP. This
    gives the model a per-user bias vector it can freely tune — capturing patterns
    that aggregate features like category preferences cannot express (e.g. two
    users with identical department distributions but completely different item
    histories will get different embeddings).

    NOTE: ID embeddings require enough interactions per user to avoid overfitting
    the embedding table. Rule of thumb: at least 100–200 interactions per user.
    With ~50k interactions / 2000 users (~25/user), use n_users=None (features only).
    With full 1.4M interactions / 2000 users (~700/user), ID embeddings help.
    """

    def __init__(self, input_dim: int, embed_dim: int = 64,
                 n_users: int | None = None, id_embed_dim: int = 32):
        super().__init__()
        self.id_embed = nn.Embedding(n_users, id_embed_dim) if n_users else None
        mlp_input = input_dim + (id_embed_dim if n_users else 0)
        self.net = nn.Sequential(
            nn.Linear(mlp_input, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.id_embed is not None and ids is not None:
            x = torch.cat([x, self.id_embed(ids)], dim=-1)
        return F.normalize(self.net(x), dim=-1)


class ItemTower(nn.Module):
    """
    Encodes an item feature vector into a 64-dim L2-normalized embedding.

    When n_items is provided, a learned nn.Embedding(n_items, id_embed_dim) table
    is concatenated with the content feature vector before the MLP. This captures
    item-level biases (popularity spikes, niche appeal) that content features and
    text embeddings cannot fully encode.

    NOTE: ID embeddings require sufficient interaction coverage per item.
    See UserTower docstring for guidance on when to enable.
    """

    def __init__(self, input_dim: int, embed_dim: int = 64,
                 n_items: int | None = None, id_embed_dim: int = 32):
        super().__init__()
        self.id_embed = nn.Embedding(n_items, id_embed_dim) if n_items else None
        mlp_input = input_dim + (id_embed_dim if n_items else 0)
        self.net = nn.Sequential(
            nn.Linear(mlp_input, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.id_embed is not None and ids is not None:
            x = torch.cat([x, self.id_embed(ids)], dim=-1)
        return F.normalize(self.net(x), dim=-1)
