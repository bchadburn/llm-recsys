"""
Two-Tower Grocery Recommendation Prototype
==========================================
Run with synthetic data (default):
    python main.py

Run with Instacart Market Basket Analysis data:
    python main.py --data-dir data/instacart/

Download Instacart data:
    kaggle competitions download -c instacart-market-basket-analysis
    unzip instacart-market-basket-analysis.zip -d data/instacart/
    # Or manually: https://www.kaggle.com/competitions/instacart-market-basket-analysis/data
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import faiss

from model import UserTower, ItemTower
from ranker import train_ranker, show_reranking_comparison
from eval import evaluate, print_eval_table

# ── Hyperparameters ────────────────────────────────────────────────────────────
EMBED_DIM   = 64
TEMPERATURE = 0.07
BATCH_SIZE  = 64
EPOCHS      = 20
LR          = 1e-3
SEED        = 42
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Training ───────────────────────────────────────────────────────────────────

def train(user_features, item_features, interactions, InteractionDataset):
    # Split by user, not randomly — random splits let the same user appear in
    # both train and val, leaking user-level patterns and underestimating val loss.
    n_users = user_features.shape[0]
    n_items = item_features.shape[0]
    rng = np.random.default_rng(SEED)
    user_perm  = rng.permutation(n_users)
    val_users  = set(user_perm[:int(0.2 * n_users)].tolist())
    train_ints = [(u, i) for u, i in interactions if u not in val_users]
    val_ints   = [(u, i) for u, i in interactions if u in val_users]

    train_ds = InteractionDataset(user_features, item_features, train_ints)
    val_ds   = InteractionDataset(user_features, item_features, val_ints)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # ID embeddings enabled: with ~1.4M interactions / 2000 users (~700/user)
    # the embedding tables have sufficient signal to generalize to held-out interactions.
    user_tower = UserTower(input_dim=user_features.shape[1], embed_dim=EMBED_DIM,
                           n_users=n_users)
    item_tower = ItemTower(input_dim=item_features.shape[1], embed_dim=EMBED_DIM,
                           n_items=n_items)
    optimizer  = torch.optim.Adam(
        list(user_tower.parameters()) + list(item_tower.parameters()), lr=LR
    )

    n_user_params = sum(p.numel() for p in user_tower.parameters())
    n_item_params = sum(p.numel() for p in item_tower.parameters())
    print("\nTraining (InfoNCE / NT-Xent loss, symmetric cross-entropy):")
    print(f"  {len(train_ints)} train interactions | {len(val_ints)} val interactions")
    print(f"  batch_size={BATCH_SIZE} | epochs={EPOCHS} | temperature={TEMPERATURE}")
    print(f"  user_tower params={n_user_params:,} | item_tower params={n_item_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        user_tower.train()
        item_tower.train()
        train_loss = 0.0

        for user_feat, item_feat, user_ids, item_ids in train_loader:
            b = user_feat.size(0)
            user_emb = user_tower(user_feat, ids=user_ids)
            item_emb = item_tower(item_feat, ids=item_ids)

            logits = (user_emb @ item_emb.T) / TEMPERATURE
            labels = torch.arange(b)
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
                for user_feat, item_feat, user_ids, item_ids in val_loader:
                    b = user_feat.size(0)
                    user_emb = user_tower(user_feat, ids=user_ids)
                    item_emb = item_tower(item_feat, ids=item_ids)
                    logits   = (user_emb @ item_emb.T) / TEMPERATURE
                    labels   = torch.arange(b)
                    val_loss += ((F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2).item()
            print(
                f"  Epoch {epoch:2d}/{EPOCHS} | "
                f"train_loss={train_loss / len(train_loader):.4f} | "
                f"val_loss={val_loss / len(val_loader):.4f}"
            )

    return user_tower, item_tower


# ── FAISS Index ────────────────────────────────────────────────────────────────

def build_faiss_index(item_tower, item_features):
    """Embed all items and load them into an exact inner-product FAISS index."""
    item_tower.eval()
    item_ids = torch.arange(len(item_features))
    with torch.no_grad():
        item_embs = item_tower(torch.tensor(item_features), ids=item_ids).numpy().astype(np.float32)
    faiss.normalize_L2(item_embs)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(item_embs)
    return index


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(user_tower, index, user_features, items, user_archetypes,
                  archetype_labels, prefs_slice, categories):
    user_tower.eval()

    demo_users = {}
    for label in archetype_labels:
        if label not in categories:
            continue
        target_idx = categories.index(label)
        matches = np.where(user_archetypes == target_idx)[0]
        if len(matches):
            demo_users[label] = int(matches[0])

    print("\n" + "=" * 72)
    print("  INFERENCE DEMO — Top-10 Recommendations via FAISS cosine similarity")
    print("=" * 72)

    for archetype, user_id in demo_users.items():
        feat = torch.tensor(user_features[user_id]).unsqueeze(0)
        uid  = torch.tensor([user_id])
        with torch.no_grad():
            user_emb = user_tower(feat, ids=uid).numpy().astype(np.float32)
        faiss.normalize_L2(user_emb)

        scores, indices = index.search(user_emb, 10)

        prefs    = user_features[user_id][prefs_slice]
        top_cats = sorted(range(len(prefs)), key=lambda i: -prefs[i])[:3]
        cat_summary = "  |  ".join(
            f"{categories[c]} ({prefs[c]*100:.0f}%)" for c in top_cats
        )

        print(f"\n  User {user_id:3d}  [archetype: {archetype}]")
        print(f"  Purchase profile: {cat_summary}")
        print()
        print(f"  {'Rank':<5} {'Item':<28} {'Category':<16} {'Cosine':<8}")
        print("  " + "-" * 60)
        for rank, (item_idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            item = items[item_idx]
            print(
                f"  {rank:<5} {item['name']:<28} {item['category']:<16} {score:.4f}"
            )


# ── Side-by-side Comparison ────────────────────────────────────────────────────

def compare_inference(user_tower_oh, index_oh, user_tower_te, index_te,
                      user_features, items, user_archetypes,
                      archetype_labels, prefs_slice, categories,
                      label_oh="one-hot features", label_te="text embeddings (384d)"):
    demo_users = {}
    for label in archetype_labels:
        if label not in categories:
            continue
        target_idx = categories.index(label)
        matches = np.where(user_archetypes == target_idx)[0]
        if len(matches):
            demo_users[label] = int(matches[0])

    print("\n" + "=" * 94)
    print(f"  SIDE-BY-SIDE: {label_oh}  vs  {label_te}")
    print("=" * 94)

    for archetype, user_id in demo_users.items():
        feat = torch.tensor(user_features[user_id]).unsqueeze(0)
        uid  = torch.tensor([user_id])

        user_tower_oh.eval()
        with torch.no_grad():
            emb_oh = user_tower_oh(feat, ids=uid).numpy().astype(np.float32)
        faiss.normalize_L2(emb_oh)
        scores_oh, idx_oh = index_oh.search(emb_oh, 10)

        user_tower_te.eval()
        with torch.no_grad():
            emb_te = user_tower_te(feat, ids=uid).numpy().astype(np.float32)
        faiss.normalize_L2(emb_te)
        scores_te, idx_te = index_te.search(emb_te, 10)

        prefs    = user_features[user_id][prefs_slice]
        top_cats = sorted(range(len(prefs)), key=lambda i: -prefs[i])[:3]
        cat_summary = "  |  ".join(f"{categories[c]} ({prefs[c]*100:.0f}%)" for c in top_cats)

        print(f"\n  User {user_id:3d}  [archetype: {archetype}]  —  {cat_summary}")
        print()
        print(f"  {'':4}  {label_oh.upper():<44}  {label_te.upper():<44}")
        print(f"  {'Rank':<4}  {'Item':<26} {'Category':<12} {'Score':<5}  "
              f"{'Item':<26} {'Category':<12} {'Score':<5}")
        print("  " + "-" * 92)
        for rank in range(10):
            a = items[idx_oh[0][rank]]
            b = items[idx_te[0][rank]]
            print(
                f"  {rank+1:<4}  {a['name']:<26} {a['category']:<12} {scores_oh[0][rank]:.3f}  "
                f"{b['name']:<26} {b['category']:<12} {scores_te[0][rank]:.3f}"
            )


# ── Production Notes ───────────────────────────────────────────────────────────

PRODUCTION_NOTES = """
================================================================================
  WHERE TO ADD COMPLEXITY IN A PRODUCTION SYSTEM
================================================================================

  1. ID EMBEDDINGS (highest priority)
     Add nn.Embedding(n_users, d) and nn.Embedding(n_items, d) tables.
     Concatenate learned ID embeddings with hand-crafted feature vectors before
     each tower MLP. This captures user/item-specific biases that feature
     vectors alone cannot express.

  2. TWO-STAGE RETRIEVAL
     FAISS handles *recall* — retrieve a candidate set (top-100 to top-500).
     A separate ranking stage (cross-encoder, LightGBM, or deep ranker)
     re-ranks candidates using expensive pairwise features that are too costly
     to compute at full catalog scale.

  3. APPROXIMATE NEAREST NEIGHBOR AT SCALE
     Above ~50k items, replace IndexFlatIP with:
       - IndexHNSWFlat : graph-based, sub-ms retrieval at >99% recall
       - IndexIVFPQ    : compressed, suits millions of items

  4. FEATURE STORE
     User features change on every purchase. A feature store (Redis, Feast,
     Tecton) serves pre-computed user vectors at <10 ms latency.

  5. HARD NEGATIVE MINING
     In-batch negatives are mostly trivially easy (random unrelated items).
     Hard negatives — items the user didn't buy but that scored highly in a
     prior retrieval pass — dramatically improve recall@K at larger scales.

  6. COLD-START FOR NEW ITEMS
     New items have no interactions. Text embeddings (sentence-transformer or
     LLM-generated descriptions) let you embed and index new items immediately.

  7. ONLINE / INCREMENTAL LEARNING
     Retrain the user tower hourly or on purchase triggers. Versioned model
     checkpoints + blue/green FAISS index swaps prevent stale embeddings.

================================================================================
"""


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Two-tower grocery recommendation prototype",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                          # synthetic data\n"
            "  python main.py --data-dir data/instacart/  # Instacart data\n"
            "  python main.py --data-dir data/instacart/ --n-users 2000 --n-items 5000"
        ),
    )
    parser.add_argument(
        '--data-dir', type=str, default=None,
        help='Path to directory containing Instacart CSV files. '
             'Uses synthetic data if omitted.',
    )
    parser.add_argument(
        '--n-users', type=int, default=5000,
        help='Max users to load from Instacart (default: 5000). Ignored for synthetic.',
    )
    parser.add_argument(
        '--n-items', type=int, default=10000,
        help='Max items to load from Instacart (default: 10000). Ignored for synthetic.',
    )
    parser.add_argument(
        '--n-interactions', type=int, default=50_000,
        help='Max training interactions sampled from Instacart (default: 50000). '
             'Features are always built from full history. Use 200000+ for best quality. '
             'Ignored for synthetic.',
    )
    args = parser.parse_args()

    # ── Data loading ───────────────────────────────────────────────────────────
    if args.data_dir:
        from data_instacart import (
            load_instacart,
            get_item_text_embeddings,
            DEPARTMENTS as CATEGORIES,
            ARCHETYPE_LABELS,
            USER_PREFS_SLICE,
            PRICE_SENS_IDX,
        )
        from data import InteractionDataset

        print("=" * 72)
        print("  TWO-TOWER GROCERY RECOMMENDATION — INSTACART DATA")
        print("=" * 72)
        print(f"\nLoading Instacart data from: {args.data_dir}")

        user_features, item_features, items, interactions, user_archetypes, up_stats = load_instacart(
            args.data_dir, n_users=args.n_users, n_items=args.n_items,
            n_interactions=args.n_interactions,
        )
        dataset_label_oh = f"Instacart features ({item_features.shape[1]}d)"
        dataset_label_te = "text embeddings (384d, aisle+dept)"
    else:
        from data import (
            generate_data,
            get_item_text_embeddings,
            InteractionDataset,
            CATEGORIES,
            ARCHETYPE_LABELS,
            USER_PREFS_SLICE,
            PRICE_SENS_IDX,
        )

        print("=" * 72)
        print("  TWO-TOWER GROCERY RECOMMENDATION PROTOTYPE — SYNTHETIC DATA")
        print("=" * 72)
        print("\nGenerating synthetic grocery data...")

        user_features, item_features, items, interactions, user_archetypes = generate_data(
            n_users=500, n_items=200, n_interactions=5000,
        )
        up_stats = None
        dataset_label_oh = "one-hot features (13d)"
        dataset_label_te = "text embeddings (384d, all-MiniLM-L6-v2)"

    print(
        f"  {len(user_features)} users  |  {len(items)} items  |  "
        f"{len(interactions)} interactions"
    )
    print(f"  User feature dim: {user_features.shape[1]}  |  Item feature dim: {item_features.shape[1]}")

    # ── Approach A: tabular item features ─────────────────────────────────────
    print(f"\n--- Approach A: {dataset_label_oh} ---")
    user_tower_oh, item_tower_oh = train(user_features, item_features, interactions, InteractionDataset)
    print("\nBuilding FAISS index (tabular)...")
    index_oh = build_faiss_index(item_tower_oh, item_features)
    print(f"  Index ready: {index_oh.ntotal} vectors, dim={EMBED_DIM}")

    # ── Approach B: sentence-transformer text embeddings (384d) ───────────────
    print(f"\n--- Approach B: {dataset_label_te} ---")
    print("Loading sentence-transformer and embedding items...")
    item_text_embs = get_item_text_embeddings(items)
    print(f"  Item text embeddings: {item_text_embs.shape}")
    user_tower_te, item_tower_te = train(user_features, item_text_embs, interactions, InteractionDataset)
    print("\nBuilding FAISS index (text embeddings)...")
    index_te = build_faiss_index(item_tower_te, item_text_embs)
    print(f"  Index ready: {index_te.ntotal} vectors, dim={EMBED_DIM}")

    # ── Side-by-side comparison ────────────────────────────────────────────────
    compare_inference(
        user_tower_oh, index_oh, user_tower_te, index_te,
        user_features, items, user_archetypes,
        archetype_labels=ARCHETYPE_LABELS,
        prefs_slice=USER_PREFS_SLICE,
        categories=CATEGORIES,
        label_oh=dataset_label_oh,
        label_te=dataset_label_te,
    )

    # ── Offline evaluation ─────────────────────────────────────────────────────
    print("\nRunning offline evaluation...")
    results_oh = evaluate(user_tower_oh, index_oh, user_features, interactions)
    results_te = evaluate(user_tower_te, index_te, user_features, interactions)
    print_eval_table(results_oh, results_te)

    # ── LightGBM ranker ────────────────────────────────────────────────────────
    ranker = train_ranker(
        user_tower_te, item_tower_te, index_te,
        user_features, item_text_embs, items, interactions,
        price_sens_idx=PRICE_SENS_IDX,
        up_stats=up_stats,
    )
    show_reranking_comparison(
        ranker, user_tower_te, item_tower_te, index_te,
        user_features, item_text_embs, items, user_archetypes,
        archetype_labels=ARCHETYPE_LABELS,
        prefs_slice=USER_PREFS_SLICE,
        price_sens_idx=PRICE_SENS_IDX,
        categories=CATEGORIES,
        up_stats=up_stats,
    )

    print(PRODUCTION_NOTES)


if __name__ == "__main__":
    main()
