"""
Experiment 5: Dual-Head Item Tower
===================================
Tests whether frozen semantic embeddings add value when the model is
*forced* to use them via a separate head, rather than being allowed to warp
them freely through a single MLP.

Hypothesis
----------
In Exp 1, text embeddings were passed as the sole item features and projected
through a single MLP trained with InfoNCE loss. The MLP had permission to
destroy the semantic structure — and it did, because collaborative signal
dominates the loss. That's why Exp 1 showed no improvement over tabular features.

This experiment tests the counterfactual: what if we freeze the semantic head
(a single linear projection that preserves cosine structure) and let the model
learn a scalar gate α to weight it against a freely-trained collaborative head?

If α → 0 after training: collaborative dominates, semantic adds nothing
If α > 0.3 and metrics improve: semantic genuinely helps at this catalog scale

Methods compared
----------------
A  Baseline        : tabular item features only (standard ItemTower)
B  Fused (frozen)  : DualHeadItemTower, semantic_head frozen after init
C  Fused (trainable): DualHeadItemTower, semantic_head allowed to train
                      (control — should converge toward Exp 1 behaviour)
"""

import numpy as np
import torch
import torch.nn.functional as F
import faiss
from pathlib import Path
from torch.utils.data import DataLoader
from dotenv import load_dotenv

load_dotenv(Path.home() / '.env')

DATA_DIR    = Path('data/instacart')
EMBED_DIM   = 64
TEMPERATURE = 0.07
BATCH_SIZE  = 64
EPOCHS      = 20
LR          = 1e-3
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    from data_instacart import load_instacart, get_item_text_embeddings
    print("Loading Instacart data...")
    user_features, item_features, items, interactions, user_archetypes, up_stats = load_instacart(
        str(DATA_DIR), n_users=2000, n_items=5000, n_interactions=1500000,
    )
    print("  Loading sentence-transformer embeddings...")
    semantic_features = get_item_text_embeddings(items)
    print(f"  Semantic features: {semantic_features.shape}")
    return user_features, item_features, semantic_features, items, interactions, user_archetypes


# ── Dataset with dual item features ──────────────────────────────────────────

class DualFeatureDataset(torch.utils.data.Dataset):
    """Returns (user_feat, semantic_item_feat, collab_item_feat, user_id, item_id)."""

    def __init__(self, user_features, semantic_features, collab_features, interactions):
        self.user_features    = torch.tensor(user_features,    dtype=torch.float32)
        self.semantic_features = torch.tensor(semantic_features, dtype=torch.float32)
        self.collab_features  = torch.tensor(collab_features,  dtype=torch.float32)
        self.interactions     = interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        uid, iid = self.interactions[idx]
        return (
            self.user_features[uid],
            self.semantic_features[iid],
            self.collab_features[iid],
            uid,
            iid,
        )


# ── Training ──────────────────────────────────────────────────────────────────

def train_baseline(user_features, item_features, interactions):
    """Approach A: standard ItemTower on tabular features."""
    from main import train, build_faiss_index
    from data import InteractionDataset
    print("  Training baseline (tabular features)...")
    user_tower, item_tower = train(user_features, item_features, interactions, InteractionDataset)
    index = build_faiss_index(item_tower, item_features)
    return user_tower, item_tower, index


def train_dual_head(user_features, semantic_features, collab_features, interactions,
                    freeze_semantic: bool, label: str):
    """
    Approaches B and C: DualHeadItemTower.

    freeze_semantic=True  → semantic head is a fixed linear projection (Approach B)
    freeze_semantic=False → semantic head trains freely (Approach C, control)
    """
    from model import UserTower, DualHeadItemTower

    n_users = user_features.shape[0]
    n_items = semantic_features.shape[0]
    semantic_dim = semantic_features.shape[1]
    collab_dim   = collab_features.shape[1]

    rng = np.random.default_rng(SEED)
    perm       = rng.permutation(len(interactions))
    split      = int(0.8 * len(interactions))
    train_ints = [interactions[i] for i in perm[:split]]
    val_ints   = [interactions[i] for i in perm[split:]]

    train_ds = DualFeatureDataset(user_features, semantic_features, collab_features, train_ints)
    val_ds   = DualFeatureDataset(user_features, semantic_features, collab_features, val_ints)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    user_tower = UserTower(
        input_dim=user_features.shape[1], embed_dim=EMBED_DIM, n_users=n_users
    ).to(DEVICE)
    item_tower = DualHeadItemTower(
        semantic_dim=semantic_dim, collab_dim=collab_dim,
        embed_dim=EMBED_DIM, n_items=n_items,
    ).to(DEVICE)

    if freeze_semantic:
        for p in item_tower.semantic_head.parameters():
            p.requires_grad = False

    optimizer = torch.optim.Adam(
        [p for p in list(user_tower.parameters()) + list(item_tower.parameters())
         if p.requires_grad],
        lr=LR,
    )

    frozen_note = " [semantic head FROZEN]" if freeze_semantic else " [semantic head trainable]"
    print(f"\n  Training {label}{frozen_note}")
    print(f"  {len(train_ints)} train | {len(val_ints)} val | device={DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        user_tower.train()
        item_tower.train()
        train_loss = 0.0

        for user_feat, sem_feat, col_feat, user_ids, item_ids in train_loader:
            user_feat = user_feat.to(DEVICE)
            sem_feat  = sem_feat.to(DEVICE)
            col_feat  = col_feat.to(DEVICE)
            user_ids  = user_ids.to(DEVICE)
            item_ids  = item_ids.to(DEVICE)
            b = user_feat.size(0)

            user_emb = user_tower(user_feat, ids=user_ids)
            item_emb = item_tower(sem_feat, col_feat, ids=item_ids)

            logits = (user_emb @ item_emb.T) / TEMPERATURE
            labels = torch.arange(b, device=DEVICE)
            loss   = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % 5 == 0:
            user_tower.eval()
            item_tower.eval()
            val_loss = 0.0
            with torch.no_grad():
                for user_feat, sem_feat, col_feat, user_ids, item_ids in val_loader:
                    user_feat = user_feat.to(DEVICE)
                    sem_feat  = sem_feat.to(DEVICE)
                    col_feat  = col_feat.to(DEVICE)
                    user_ids  = user_ids.to(DEVICE)
                    item_ids  = item_ids.to(DEVICE)
                    b = user_feat.size(0)
                    user_emb = user_tower(user_feat, ids=user_ids)
                    item_emb = item_tower(sem_feat, col_feat, ids=item_ids)
                    logits   = (user_emb @ item_emb.T) / TEMPERATURE
                    labels   = torch.arange(b, device=DEVICE)
                    val_loss += ((F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2).item()
            print(
                f"  Epoch {epoch:2d}/{EPOCHS} | "
                f"train={train_loss/len(train_loader):.4f} | "
                f"val={val_loss/len(val_loader):.4f} | "
                f"α(semantic)={item_tower.alpha:.3f}"
            )

    print(f"\n  Final α (semantic weight): {item_tower.alpha:.4f}")
    print(f"  Interpretation: {'semantic adds meaningful weight' if item_tower.alpha > 0.35 else 'collaborative dominates — semantic marginally useful' if item_tower.alpha > 0.2 else 'semantic largely ignored'}")

    # Build FAISS index using the dual-head tower
    item_tower.eval()
    sem_t = torch.tensor(semantic_features, dtype=torch.float32, device=DEVICE)
    col_t = torch.tensor(collab_features,   dtype=torch.float32, device=DEVICE)
    ids_t = torch.arange(n_items, device=DEVICE)
    with torch.no_grad():
        item_embs = item_tower(sem_t, col_t, ids=ids_t).cpu().numpy().astype(np.float32)
    faiss.normalize_L2(item_embs)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(item_embs)

    return user_tower, item_tower, index


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_tower(user_tower, index, user_features, interactions, ks=(5, 10, 20)):
    from eval import evaluate
    return evaluate(user_tower, index, user_features, interactions, ks=ks)


def print_summary(results: dict, ks=(5, 10, 20)):
    ks = sorted(ks)
    print("\n" + "="*72)
    print("  EXPERIMENT 5 SUMMARY — Dual-Head Item Tower")
    print("="*72)
    print(f"\n  {'Method':<35} {'R@5':<10} {'R@10':<10} {'R@20':<10} {'N@10':<10}")
    print("  " + "-"*65)
    baseline = results['baseline']['recall'][20]
    for label, res in results.items():
        r5  = res['recall'][5]
        r10 = res['recall'][10]
        r20 = res['recall'][20]
        n10 = res['ndcg'][10]
        delta = f"  ({r20-baseline:+.4f} vs baseline)" if label != 'baseline' else ""
        print(f"  {label:<35} {r5:<10.4f} {r10:<10.4f} {r20:<10.4f} {n10:<10.4f}{delta}")

    print(f"\n  Users evaluated: {results['baseline']['n_eval_users']}")
    print()
    print("  Interpretation guide:")
    print("  · α printed during training = learned semantic weight (0=collab only, 1=semantic only)")
    print("  · If Approach B (frozen semantic) > baseline: semantic info helps when preserved")
    print("  · If Approach C (trainable) ≈ baseline: confirms the MLP warps semantics away")
    print("  · If B ≈ C: semantic projection shape doesn't matter, only collaborative signal does")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*72)
    print("  EXPERIMENT 5: DUAL-HEAD ITEM TOWER")
    print("="*72)
    print("  Testing whether frozen semantic embeddings add value when forced")
    print("  to contribute via a separate head with a learned gate weight (α).")

    user_features, collab_features, semantic_features, items, interactions, _ = load_data()

    results = {}

    # Approach A — baseline tabular
    user_tower_a, _, index_a = train_baseline(user_features, collab_features, interactions)
    results['baseline (tabular)'] = evaluate_tower(user_tower_a, index_a, user_features, interactions)

    # Approach B — frozen semantic head (the key test)
    user_tower_b, item_tower_b, index_b = train_dual_head(
        user_features, semantic_features, collab_features, interactions,
        freeze_semantic=True, label="Approach B (frozen semantic)",
    )
    results['dual-head frozen semantic'] = evaluate_tower(user_tower_b, index_b, user_features, interactions)

    # Approach C — trainable semantic head (control: should ≈ Exp 1)
    user_tower_c, item_tower_c, index_c = train_dual_head(
        user_features, semantic_features, collab_features, interactions,
        freeze_semantic=False, label="Approach C (trainable semantic)",
    )
    results['dual-head trainable semantic'] = evaluate_tower(user_tower_c, index_c, user_features, interactions)

    print_summary(results)


if __name__ == '__main__':
    main()
