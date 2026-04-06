"""
xgb_model.py — XGBoost direct ranking model (Approach C).

# =============================================================================
# WHY XGBoost AS A DIRECT RANKER?
# =============================================================================
#
# Top Kaggle competitors on Instacart skipped neural retrieval entirely and
# trained a single GBDT classifier directly on (user, item) feature pairs.
# This works well here because:
#
#   1. Repeat purchases dominate (~60% of orders are reorders).  UP interaction
#      features (reorder rate, days since last purchase, etc.) are highly
#      predictive on their own — no embedding needed.
#
#   2. User/item IDs as ordinal features give the tree the same memorization
#      power as ID embeddings in a neural model, without a separate embedding
#      table or training stage.
#
#   3. Tree splits handle high-cardinality categoricals and non-linear
#      feature interactions natively.
#
# Candidate generation:
#   For tractability we limit evaluation to:
#     - All items the user has purchased before (prior order items)
#     - Top-N globally popular items (cold candidates)
#   This covers the vast majority of actual purchases at low compute cost.
#
# Feature layout (per (user, item) pair):
#   [0]        user_id          — ordinal; tree splits learn per-user biases
#   [1]        item_id          — ordinal; tree splits learn per-item biases
#   [2:26]     user_features    — 24-d hand-crafted user feature vector
#   [26:50]    item_features    — 24-d hand-crafted item feature vector
#   [50:57]    up_stats         — 7-d user×product interaction features
#                                 (zeros if pair unseen in training history)
# =============================================================================
"""

from datetime import datetime
import numpy as np
import xgboost as xgb
from collections import defaultdict


_UP_ZEROS = np.zeros(7, dtype=np.float32)

N_POPULAR  = 200   # number of globally popular cold candidates per user
EVAL_K     = (5, 10, 20)


def _build_features(user_id: int, item_id: int,
                    user_features: np.ndarray, item_features: np.ndarray,
                    up_stats: dict) -> np.ndarray:
    up = up_stats.get((user_id, item_id), _UP_ZEROS)
    return np.concatenate([
        [float(user_id), float(item_id)],
        user_features[user_id].astype(np.float32),
        item_features[item_id].astype(np.float32),
        up,
    ])


def _popular_items(interactions: list, n: int) -> list[int]:
    counts: dict[int, int] = defaultdict(int)
    for _, item_id in interactions:
        counts[item_id] += 1
    return sorted(counts, key=lambda i: -counts[i])[:n]


def train_xgb(user_features: np.ndarray, item_features: np.ndarray,
              interactions: list, up_stats: dict,
              seed: int = 42) -> xgb.XGBClassifier:
    """
    Train an XGBoost binary classifier on (user, item) pairs.

    Positive labels: interactions in the training split.
    Negative labels: popular items the user did NOT interact with.
    Negatives are sampled at 4:1 ratio to keep training tractable.

    Returns a trained XGBClassifier.
    """
    rng = np.random.default_rng(seed)
    perm       = rng.permutation(len(interactions))
    split      = int(0.8 * len(interactions))
    train_ints = [interactions[i] for i in perm[:split]]

    # Build per-user positive set from training interactions
    user_positives: dict[int, set] = defaultdict(set)
    for u, i in train_ints:
        user_positives[u].add(i)

    popular = _popular_items(train_ints, N_POPULAR)

    rows, labels = [], []
    for u, pos_items in user_positives.items():
        # Positive examples
        for i in pos_items:
            rows.append(_build_features(u, i, user_features, item_features, up_stats))
            labels.append(1)

        # Negative examples: popular items not purchased by this user
        neg_pool = [i for i in popular if i not in pos_items]
        n_neg    = min(len(neg_pool), 4 * len(pos_items))
        neg_idx  = rng.choice(len(neg_pool), size=n_neg, replace=False)
        for idx in neg_idx:
            rows.append(_build_features(u, neg_pool[idx], user_features, item_features, up_stats))
            labels.append(0)

    X = np.vstack(rows).astype(np.float32)
    y = np.array(labels, dtype=np.int32)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    feat_names = (
        ["user_id", "item_id"]
        + [f"user_feat_{i}" for i in range(user_features.shape[1])]
        + [f"item_feat_{i}" for i in range(item_features.shape[1])]
        + ["up_purchase_count_norm", "up_reorder_rate", "up_user_order_frac",
           "up_days_since_last_order", "up_orders_since_last_order",
           "up_order_streak_norm", "up_order_rate"]
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=n_neg / max(n_pos, 1),
        objective="binary:logistic",
        eval_metric="logloss",
        seed=seed,
        verbosity=0,
    )

    print("\nTraining XGBoost direct ranker...")
    print(f"  Train pairs    : {len(X):,}  ({n_pos:,} pos / {n_neg:,} neg)")
    print(f"  Feature dim    : {X.shape[1]}")
    t0 = datetime.now()
    model.fit(X, y)
    print(f"  XGBoost training complete. ({datetime.now() - t0})")
    return model


def evaluate_xgb(model: xgb.XGBClassifier,
                 user_features: np.ndarray, item_features: np.ndarray,
                 interactions: list, up_stats: dict,
                 seed: int = 42, ks: tuple = EVAL_K) -> dict:
    """
    Evaluate XGBoost ranker on held-out val interactions.

    Candidate set per user: all items the user purchased (prior history) +
    top-N globally popular items. This mirrors how Kaggle submissions generated
    their candidate sets.

    Returns the same dict format as eval.evaluate() for easy comparison.
    """
    rng = np.random.default_rng(seed)
    perm       = rng.permutation(len(interactions))
    split      = int(0.8 * len(interactions))
    train_ints = [interactions[i] for i in perm[:split]]
    val_ints   = [interactions[i] for i in perm[split:]]

    # Ground truth: val purchases per user
    val_purchases: dict[int, set] = defaultdict(set)
    for u, i in val_ints:
        val_purchases[u].add(i)

    # Candidate pool per user: training history + popular items
    train_history: dict[int, set] = defaultdict(set)
    for u, i in train_ints:
        train_history[u].add(i)

    popular = _popular_items(train_ints, N_POPULAR)

    max_k = max(ks)
    recall_scores: dict[int, list] = {k: [] for k in ks}
    ndcg_scores:   dict[int, list] = {k: [] for k in ks}

    for user_id, relevant in val_purchases.items():
        candidates = list(train_history[user_id] | set(popular))

        feat_rows = np.vstack([
            _build_features(user_id, i, user_features, item_features, up_stats)
            for i in candidates
        ]).astype(np.float32)
        scores   = model.predict_proba(feat_rows)[:, 1]
        order    = np.argsort(-scores)
        ranked   = [candidates[j] for j in order[:max_k]]

        for k in ks:
            top_k = ranked[:k]
            hits  = sum(1 for i in top_k if i in relevant)
            recall_scores[k].append(hits / len(relevant))

            dcg  = sum(1.0 / np.log2(r + 2) for r, i in enumerate(top_k) if i in relevant)
            idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
            ndcg_scores[k].append(dcg / idcg if idcg > 0 else 0.0)

    return {
        'recall':       {k: float(np.mean(v)) for k, v in recall_scores.items()},
        'ndcg':         {k: float(np.mean(v)) for k, v in ndcg_scores.items()},
        'n_eval_users': len(val_purchases),
    }


def print_xgb_eval(results: dict, ks: tuple = EVAL_K) -> None:
    ks = sorted(ks)
    col_w    = 9
    k_header = "".join(f"{'K='+str(k):<{col_w}}" for k in ks)
    sep      = "  " + "-" * (30 + len(ks) * col_w)

    print("\n" + "=" * 72)
    print("  OFFLINE EVALUATION — XGBoost direct ranker (Approach C)")
    print("=" * 72)
    print(f"  Users evaluated : {results['n_eval_users']}")
    print()
    print(f"  {'Metric':<30}{k_header}")
    print(sep)
    for metric, key in [("Recall", "recall"), ("NDCG", "ndcg")]:
        row = "".join(f"{results[key][k]:<{col_w}.4f}" for k in ks)
        print(f"  {metric:<8} [XGBoost (user_id+item_id)]  {row}")
    print(sep)
