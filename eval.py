"""
eval.py — Offline evaluation for the two-tower grocery recommendation system.

# =============================================================================
# RECALL@K vs NDCG@K: which metric measures what?
# =============================================================================
#
# Recall@K — the retrieval metric
#   "Of all items this user actually purchased, what fraction appear in the
#   top-K results?"  Value = |retrieved_K ∩ purchased| / |purchased|.
#
#   Why it matters for retrieval: the two-tower + FAISS is a *recall* stage.
#   Its only job is to make sure relevant items are somewhere in the candidate
#   set before the ranker sees it.  Downstream stages (ranker.py) can reorder
#   candidates but cannot surface an item that was never retrieved.  Recall@20
#   answers: "did we even find the right things?"  A low Recall@20 means the
#   ranker is working with an impoverished candidate set — no amount of ranking
#   sophistication can recover relevance that was lost at retrieval.
#
# NDCG@K — the ranking metric
#   Normalized Discounted Cumulative Gain rewards placing relevant items
#   higher in the ranked list.  Relevance at rank 1 contributes 1/log2(2)=1.0;
#   at rank K it contributes 1/log2(K+1).  Normalised by IDCG (the ideal
#   ordering) so 1.0 is a perfect ranking and 0.0 means all relevant items
#   are at the bottom.
#
#   Why it matters for ranking: NDCG distinguishes a model that surfaces the
#   best-match item at rank 1 from one that buries it at rank 10.  Two models
#   can have identical Recall@10 while one has far higher NDCG@10 because it
#   consistently places the most relevant items first.  Use Recall@K to gate
#   the retrieval stage; use NDCG@K to evaluate the ranking stage.
#
# Rule of thumb:
#   Retrieval stage → optimise Recall@K (did we include the right candidates?)
#   Ranking  stage  → optimise NDCG@K   (did we order the candidates well?)
# =============================================================================
"""

import numpy as np
import torch
import faiss


# ── Val-split reconstruction ──────────────────────────────────────────────────

def _reconstruct_val_interactions(interactions: list, n_users: int, seed: int = 42) -> list:
    """
    Reproduce the user-level 20% held-out split from train() in main.py.

    Splits by user ID (not randomly by interaction) to prevent the same user
    from appearing in both train and val — which would leak user-level patterns
    and make val loss look better than it is on truly unseen users.
    """
    rng = np.random.default_rng(seed)
    user_perm = rng.permutation(n_users)
    val_users = set(user_perm[:int(0.2 * n_users)].tolist())
    return [(u, i) for u, i in interactions if u in val_users]


# ── Per-query metric helpers ──────────────────────────────────────────────────

def _recall_at_k(retrieved: np.ndarray, relevant: set, k: int) -> float:
    hits = sum(1 for item_id in retrieved[:k] if int(item_id) in relevant)
    return hits / len(relevant)


def _ndcg_at_k(retrieved: np.ndarray, relevant: set, k: int) -> float:
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item_id in enumerate(retrieved[:k])
        if int(item_id) in relevant
    )
    n_rel = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


# ── Main evaluation entry point ───────────────────────────────────────────────

def evaluate(
    user_tower,
    index,
    user_features: np.ndarray,
    interactions: list,
    ks: tuple = (5, 10, 20),
    seed: int = 42,
    n_users: int = None,
) -> dict:
    """
    Compute mean Recall@K and NDCG@K over the held-out val interactions.

    Reproduces the same 20%/80% random split used inside train() so the
    evaluation set is genuinely held out from both approaches.

    Returns:
        {
            'recall':       {5: float, 10: float, 20: float},
            'ndcg':         {5: float, 10: float, 20: float},
            'n_eval_users': int,
        }
    """
    n_users = n_users or user_features.shape[0]
    val_interactions = _reconstruct_val_interactions(interactions, n_users=n_users, seed=seed)

    # Build per-user ground-truth purchase sets from val interactions only
    val_purchases: dict[int, set] = {}
    for uid, iid in val_interactions:
        val_purchases.setdefault(uid, set()).add(iid)

    # Embed all users once (pass contiguous IDs so ID embedding tables are used if present)
    user_tower.eval()
    all_user_ids = torch.arange(n_users)
    with torch.no_grad():
        all_user_embs = user_tower(torch.tensor(user_features), ids=all_user_ids).numpy().astype(np.float32)
    faiss.normalize_L2(all_user_embs)

    max_k = max(ks)
    recall_scores: dict[int, list] = {k: [] for k in ks}
    ndcg_scores:   dict[int, list] = {k: [] for k in ks}

    for user_id, relevant in val_purchases.items():
        query = all_user_embs[user_id:user_id + 1]
        _, indices = index.search(query, max_k)
        retrieved = indices[0]

        for k in ks:
            recall_scores[k].append(_recall_at_k(retrieved, relevant, k))
            ndcg_scores[k].append(_ndcg_at_k(retrieved, relevant, k))

    return {
        'recall':       {k: float(np.mean(v)) for k, v in recall_scores.items()},
        'ndcg':         {k: float(np.mean(v)) for k, v in ndcg_scores.items()},
        'n_eval_users': len(val_purchases),
    }


# ── Comparison table ──────────────────────────────────────────────────────────

def print_eval_table(
    results_oh: dict,
    results_te: dict,
    ks: tuple = (5, 10, 20),
) -> None:
    """Print a side-by-side evaluation table for one-hot vs text-embedding approaches."""
    ks = sorted(ks)
    col_w = 9
    k_header = "".join(f"{'K='+str(k):<{col_w}}" for k in ks)

    sep = "  " + "-" * (30 + len(ks) * col_w)

    print("\n" + "=" * 72)
    print("  OFFLINE EVALUATION — held-out val interactions (20% split, SEED=42)")
    print("=" * 72)
    print(f"  Users evaluated : {results_oh['n_eval_users']}")
    print()
    print(f"  {'Metric':<30}{k_header}")
    print(sep)

    for label, results in [("one-hot  (13d)", results_oh), ("text-emb (384d)", results_te)]:
        for metric, key in [("Recall", "recall"), ("NDCG", "ndcg")]:
            row = "".join(f"{results[key][k]:<{col_w}.4f}" for k in ks)
            print(f"  {metric:<8} [{label}]     {row}")
        print(sep)
