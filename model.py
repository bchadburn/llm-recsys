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


class DualHeadItemTower(nn.Module):
    """
    Item tower with separate semantic and collaborative heads, combined via a
    learned scalar gate.

    Motivation
    ----------
    A standard ItemTower projects text embeddings (384d) through a single MLP
    trained with InfoNCE loss. Because collaborative signal dominates — the loss
    only cares about predicting which items a user bought — the MLP learns to
    warp the semantic space into whatever 64d structure best predicts purchases.
    The semantic structure is not preserved; the fancy embeddings become dead weight.

    Architecture
    ------------
    Two independent sub-towers project into the same embed_dim space:

        semantic_head  : frozen 384-d text embedding → Linear(384, embed_dim)
                         No ReLU — keeps the projection linear to preserve cosine
                         relationships from the sentence-transformer space.

        collab_head    : tabular item features (+ optional ID embedding) →
                         MLP(input_dim [+ id_dim], 64, embed_dim)
                         Trained freely; learns the collaborative structure.

    A learned scalar α = sigmoid(gate) ∈ (0,1) interpolates between them:

        item_emb = L2_norm( α * semantic_out + (1 - α) * collab_out )

    Initialised at gate=0 → α=0.5 so both heads contribute equally at the start.
    The model learns to weight them based on which provides more useful signal.
    Inspecting α after training tells you directly how much the semantic space
    helped — if α → 0, collaborative dominated; if α → 0.5+, semantic added value.

    What this tests
    ---------------
    Previous experiments (Exp 1) passed text embeddings as the sole item features.
    The MLP had "permission to ignore" semantic structure by warping it freely.
    This architecture forces the model to USE semantic structure (via the frozen
    semantic head) while still learning collaborative patterns (via the collab head).
    The gate measures whether semantic information actually helps when it cannot
    be destroyed by the optimizer.
    """

    def __init__(
        self,
        semantic_dim: int,          # 384 for all-MiniLM-L6-v2
        collab_dim: int,            # tabular item feature dim
        embed_dim: int = 64,
        n_items: int | None = None,
        id_embed_dim: int = 32,
    ):
        super().__init__()

        # Semantic head: linear projection only — preserves cosine structure
        self.semantic_head = nn.Linear(semantic_dim, embed_dim, bias=False)

        # Collaborative head: MLP on tabular features + optional ID embedding
        self.id_embed = nn.Embedding(n_items, id_embed_dim) if n_items else None
        collab_input = collab_dim + (id_embed_dim if n_items else 0)
        self.collab_head = nn.Sequential(
            nn.Linear(collab_input, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, embed_dim),
        )

        # Learned interpolation gate; init=0 → α=sigmoid(0)=0.5
        self.gate = nn.Parameter(torch.zeros(1))

    @property
    def alpha(self) -> float:
        """Semantic weight in [0, 1]. Log this after training to interpret results."""
        return float(torch.sigmoid(self.gate).item())

    def forward(
        self,
        semantic_x: torch.Tensor,   # [B, semantic_dim] — frozen text embeddings
        collab_x: torch.Tensor,     # [B, collab_dim]   — tabular features
        ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sem_out = self.semantic_head(semantic_x)           # [B, embed_dim]

        if self.id_embed is not None and ids is not None:
            collab_x = torch.cat([collab_x, self.id_embed(ids)], dim=-1)
        col_out = self.collab_head(collab_x)               # [B, embed_dim]

        alpha = torch.sigmoid(self.gate)
        fused = alpha * sem_out + (1.0 - alpha) * col_out  # [B, embed_dim]
        return F.normalize(fused, dim=-1)
