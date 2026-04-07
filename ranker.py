"""
ranker.py — LightGBM re-ranking stage for the two-tower grocery system.

# =============================================================================
# WHY TWO-STAGE RETRIEVAL → RANKING?
# =============================================================================
#
# STAGE 1 — Retrieval (two-tower + FAISS)
#   The two-tower score is exactly score(u, i) = user_emb · item_emb.  Because
#   each tower is computed independently, the model can only capture signals
#   that "pass through" the dot product.  Features that require knowing both
#   the user and the item simultaneously — e.g. "does *this* user's price
#   sensitivity match *this* item's tier?" — are invisible to it.  In exchange,
#   the factorized form enables sub-millisecond ANN search over the entire item
#   catalog via FAISS, making full-catalog retrieval tractable at scale.
#
# STAGE 2 — Ranking (LightGBM lambdarank)
#   After FAISS narrows the catalog to ~20 candidates, the ranker can afford
#   to compute richer pairwise features built from both sides jointly:
#
#     • user_emb ⊕ item_emb    — full 128-d joint context; the GBDT can find
#                                 cross-dimensional interactions the dot product
#                                 collapses.
#     • dot_score              — the retrieval signal, carried forward as a
#                                 feature so the ranker can re-weight it.
#     • item popularity        — a global signal that is hard to disentangle
#                                 from category preference inside a dot product.
#     • price_tier             — a categorical feature; tree splits handle
#                                 this cleanly without needing a continuous proxy.
#     • user_price_sens × price_tier — explicit cross-feature interaction.
#                                 A budget-sensitive user should be penalised for
#                                 premium items; a price-insensitive user should
#                                 not.  The two towers encode these independently
#                                 and cannot express the interaction.
#
#   GBDT also models arbitrary non-linearities via tree splits, removing the
#   need to hand-engineer polynomial features.
#
#   Net result: retrieval maximises recall (did we fetch the right items?);
#   ranking maximises precision (are the right items at the top of the list?).
# =============================================================================
"""

import numpy as np
import torch
import faiss
import lightgbm as lgb

from data import CATEGORIES, PRICE_TIERS, PRICE_SENS_IDX as _DEFAULT_PRICE_SENS_IDX

RETRIEVAL_K = 20   # candidates fetched from FAISS per query
FINAL_K     = 10   # items shown in the side-by-side comparison


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _embed_users(user_tower, user_features: np.ndarray) -> np.ndarray:
    user_tower.eval()
    user_ids = torch.arange(len(user_features))
    with torch.no_grad():
        embs = user_tower(torch.tensor(user_features), ids=user_ids).cpu().numpy().astype(np.float32)
    faiss.normalize_L2(embs)
    return embs


def _embed_items(item_tower, item_features: np.ndarray) -> np.ndarray:
    item_tower.eval()
    item_ids = torch.arange(len(item_features))
    with torch.no_grad():
        embs = item_tower(torch.tensor(item_features), ids=item_ids).cpu().numpy().astype(np.float32)
    faiss.normalize_L2(embs)
    return embs


# ── Feature construction ──────────────────────────────────────────────────────

_UP_ZEROS = np.zeros(7, dtype=np.float32)  # fallback when up_stats is None or pair unseen


def _make_features(user_emb_1d, item_embs, faiss_scores, candidate_indices,
                   user_price_sens, items, user_id=None, up_stats=None):
    """
    Build a feature matrix for one query user.

    Feature layout (132d without UP stats, 139d with):
        [0:64]   user_emb               — 64-d user tower embedding
        [64:128] item_emb               — 64-d item tower embedding
        [128]    dot_score              — FAISS inner-product score
        [129]    popularity             — item popularity (from items dict)
        [130]    price_tier             — 0=budget / 1=mid / 2=premium
        [131]    price_interaction      — user_price_sens * price_tier
        [132]    up_purchase_count_norm
        [133]    up_reorder_rate
        [134]    up_user_order_frac
        [135]    up_days_since_last_order
        [136]    up_orders_since_last_order
        [137]    up_order_streak_norm
        [138]    up_order_rate

    The last 7 features are only present when up_stats is not None.
    """
    use_up = up_stats is not None and user_id is not None
    n_cols = 139 if use_up else 132
    n      = len(candidate_indices)
    X      = np.zeros((n, n_cols), dtype=np.float32)

    for i, (idx, score) in enumerate(zip(candidate_indices, faiss_scores)):
        item = items[idx]
        X[i, 0:64]   = user_emb_1d
        X[i, 64:128] = item_embs[idx]
        X[i, 128]    = score
        X[i, 129]    = float(item['popularity'])
        X[i, 130]    = float(item['price_tier'])
        X[i, 131]    = user_price_sens * float(item['price_tier'])
        if use_up:
            X[i, 132:139] = up_stats.get((user_id, int(idx)), _UP_ZEROS)
    return X


# ── Training ──────────────────────────────────────────────────────────────────

def train_ranker(user_tower, item_tower, index, user_features, item_features,
                 items, interactions, n_train_users: int = 200,
                 price_sens_idx: int = _DEFAULT_PRICE_SENS_IDX,
                 up_stats: dict | None = None) -> lgb.Booster:
    """
    Train a LightGBM lambdarank re-ranker on synthetic relevance labels.

    Relevance labels:
        1  — user actually purchased the candidate item
        0  — user did not purchase the candidate item

    For each training user the ranker sees exactly RETRIEVAL_K rows (one per
    FAISS candidate), grouped together so lambdarank can optimise NDCG within
    that per-user candidate list.

    Returns a trained lgb.Booster.
    """
    user_embs = _embed_users(user_tower, user_features)
    item_embs = _embed_items(item_tower, item_features)

    # Purchase lookup: user_id -> set of purchased item_ids
    purchased: dict[int, set] = {}
    for u, i in interactions:
        purchased.setdefault(u, set()).add(i)

    eligible = [u for u in range(len(user_features)) if u in purchased]
    rng = np.random.default_rng(42)
    train_users = rng.choice(
        eligible, size=min(n_train_users, len(eligible)), replace=False
    )

    feat_rows, label_rows, groups = [], [], []
    for user_id in train_users:
        query_emb = user_embs[user_id:user_id + 1]          # [1, 64]
        scores, indices = index.search(query_emb, RETRIEVAL_K)
        cands = indices[0]
        scs   = scores[0]

        labels = np.array(
            [1.0 if int(c) in purchased[user_id] else 0.0 for c in cands],
            dtype=np.float32,
        )
        price_sens = float(user_features[user_id][price_sens_idx])
        feats = _make_features(user_embs[user_id], item_embs, scs, cands, price_sens, items,
                               user_id=int(user_id), up_stats=up_stats)

        feat_rows.append(feats)
        label_rows.append(labels)
        groups.append(RETRIEVAL_K)

    X = np.vstack(feat_rows)
    y = np.concatenate(label_rows)

    feat_names = (
        [f"user_emb_{i}" for i in range(64)]
        + [f"item_emb_{i}" for i in range(64)]
        + ["dot_score", "popularity", "price_tier", "price_interaction"]
        + (["up_purchase_count_norm", "up_reorder_rate", "up_user_order_frac",
             "up_days_since_last_order", "up_orders_since_last_order",
             "up_order_streak_norm", "up_order_rate"] if up_stats else [])
    )

    train_data = lgb.Dataset(
        X, label=y, group=groups, feature_name=feat_names, free_raw_data=False
    )

    params = {
        "objective":         "lambdarank",
        "metric":            "ndcg",
        "ndcg_eval_at":      [10],
        "num_leaves":        31,
        "learning_rate":     0.05,
        "min_child_samples": 5,
        "verbose":           -1,
        "seed":              42,
    }

    print("\nTraining LightGBM lambdarank ranker...")
    print(f"  Train users    : {len(train_users)}")
    print(f"  Candidates/user: {RETRIEVAL_K}")
    up_note = " (incl. 3 UP features)" if up_stats else ""
    print(f"  Feature dim    : {X.shape[1]}{up_note}  |  Total (u,i) pairs: {X.shape[0]}")
    booster = lgb.train(params, train_data, num_boost_round=100)
    print("  Ranker training complete.")
    return booster


# ── Inference helpers ─────────────────────────────────────────────────────────

def _score_candidates(booster, user_emb_1d, item_embs, index, query_emb,
                      user_price_sens, items, user_id=None, up_stats=None):
    """
    Retrieve RETRIEVAL_K candidates then re-rank them.

    Returns:
        before : list[(item_idx, faiss_score)]  — original FAISS order
        after  : list[(item_idx, ranker_score)] — ranker order (descending)
    """
    scores, indices = index.search(query_emb, RETRIEVAL_K)
    cands = indices[0]
    scs   = scores[0]

    before = [(int(c), float(s)) for c, s in zip(cands, scs)]

    feats = _make_features(user_emb_1d, item_embs, scs, cands, user_price_sens, items,
                           user_id=user_id, up_stats=up_stats)
    ranker_scores = booster.predict(feats)
    order = np.argsort(-ranker_scores)
    after = [(int(cands[i]), float(ranker_scores[i])) for i in order]

    return before, after


# ── Side-by-side display ──────────────────────────────────────────────────────

def show_reranking_comparison(booster, user_tower, item_tower, index,
                              user_features, item_features, items, user_archetypes,
                              archetype_labels=None, prefs_slice=slice(8, 16),
                              price_sens_idx: int = _DEFAULT_PRICE_SENS_IDX,
                              categories=None, up_stats=None):
    """Print FAISS top-10 vs LightGBM re-ranked top-10 for three demo users."""
    user_embs = _embed_users(user_tower, user_features)
    item_embs = _embed_items(item_tower, item_features)

    if archetype_labels is None:
        archetype_labels = ['produce', 'snacks', 'cleaning']

    demo_users = {}
    for label in archetype_labels:
        if label not in CATEGORIES:
            continue
        target_idx = CATEGORIES.index(label)
        matches = np.where(user_archetypes == target_idx)[0]
        if len(matches):
            demo_users[label] = int(matches[0])

    print("\n" + "=" * 102)
    print("  RANKING STAGE — FAISS top-10  vs  LightGBM lambdarank top-10")
    print("=" * 102)

    for archetype, user_id in demo_users.items():
        user_emb_1d     = user_embs[user_id]
        query_emb       = user_embs[user_id:user_id + 1]
        user_price_sens = float(user_features[user_id][price_sens_idx])

        before, after = _score_candidates(
            booster, user_emb_1d, item_embs, index, query_emb, user_price_sens, items,
            user_id=user_id, up_stats=up_stats,
        )

        prefs      = user_features[user_id][prefs_slice]
        top_cats   = sorted(range(len(prefs)), key=lambda i: -prefs[i])[:3]
        cat_labels = categories if categories is not None else CATEGORIES
        cat_summary = "  |  ".join(
            f"{cat_labels[c] if c < len(cat_labels) else c} ({prefs[c] * 100:.0f}%)"
            for c in top_cats
        )

        print(f"\n  User {user_id:3d}  [archetype: {archetype}]  —  {cat_summary}")
        print(f"  Price sensitivity: {user_price_sens:.2f}")
        print()
        print(
            f"  {'Rank':<5}  {'── FAISS (cosine sim) ──':^46}"
            f"  {'── LightGBM re-ranked ──':^46}"
        )
        print(
            f"  {'':5}  {'Item':<26} {'Category':<12} {'Sim':<7}"
            f"  {'Item':<26} {'Category':<12} {'Logit':<7}"
        )
        print("  " + "-" * 104)

        for rank in range(FINAL_K):
            b_idx, b_score = before[rank]
            a_idx, a_score = after[rank]
            b_item = items[b_idx]
            a_item = items[a_idx]
            print(
                f"  {rank + 1:<5}  {b_item['name']:<26} {b_item['category']:<12} {b_score:>6.3f} "
                f" {a_item['name']:<26} {a_item['category']:<12} {a_score:>6.3f}"
            )
