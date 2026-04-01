import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """
    Encodes a 20-dim user feature vector into a 64-dim L2-normalized embedding.

    Layout of input features (20 floats):
        [0:8]   norm_counts   — log1p-normalized purchase counts per category
        [8:16]  prefs         — fraction of purchases per category (sums to 1)
        [16]    total_inter   — normalized total interaction count
        [17]    recency       — synthetic recency score in [0, 1]
        [18]    price_sens    — price sensitivity in [0, 1]
        [19]    variety       — normalized Shannon entropy of purchases
    """

    def __init__(self, input_dim: int = 20, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ItemTower(nn.Module):
    """
    Encodes an item feature vector into a 64-dim L2-normalized embedding.

    Default input is a 384-dim sentence-transformer embedding
    (all-MiniLM-L6-v2). Pass input_dim=13 to use the legacy one-hot features:
        [0:8]   category_onehot — one-hot for 8 grocery categories
        [8:11]  price_onehot    — one-hot: [budget, mid, premium]
        [11]    popularity      — log1p-normalized interaction count in [0, 1]
        [12]    avg_rating      — normalized rating in [0, 1]
    """

    def __init__(self, input_dim: int = 384, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)
