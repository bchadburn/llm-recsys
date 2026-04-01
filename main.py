"""
Two-Tower Grocery Recommendation Prototype
==========================================
Run:  python main.py

Trains user + item towers on synthetic grocery interaction data, builds a FAISS
index over all item embeddings, then retrieves the top-10 recommended items for
three different user archetypes using cosine similarity.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import faiss

from data import generate_data, InteractionDataset, CATEGORIES, PRICE_TIERS
from model import UserTower, ItemTower

# ── Hyperparameters ────────────────────────────────────────────────────────────
EMBED_DIM   = 64
TEMPERATURE = 0.07   # scales logits before BCE; lower = sharper distribution
BATCH_SIZE  = 64
EPOCHS      = 20
LR          = 1e-3
SEED        = 42
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Training ───────────────────────────────────────────────────────────────────

def train(user_features, item_features, interactions):
    dataset = InteractionDataset(user_features, item_features, interactions)
    val_size   = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    user_tower = UserTower(input_dim=user_features.shape[1], embed_dim=EMBED_DIM)
    item_tower = ItemTower(input_dim=item_features.shape[1], embed_dim=EMBED_DIM)
    optimizer  = torch.optim.Adam(
        list(user_tower.parameters()) + list(item_tower.parameters()), lr=LR
    )

    print("\nTraining (InfoNCE / NT-Xent loss, symmetric cross-entropy):")
    print(f"  {train_size} train interactions | {val_size} val interactions")
    print(f"  batch_size={BATCH_SIZE} | epochs={EPOCHS} | temperature={TEMPERATURE}\n")

    for epoch in range(1, EPOCHS + 1):
        user_tower.train()
        item_tower.train()
        train_loss = 0.0

        for user_feat, item_feat in train_loader:
            b = user_feat.size(0)
            user_emb = user_tower(user_feat)   # [B, 64], L2-normalized
            item_emb = item_tower(item_feat)   # [B, 64], L2-normalized

            # InfoNCE (NT-Xent): for each user, softmax over all items in batch.
            # Symmetric: also applied from the item side. Diagonal = positive pairs.
            # This drives positive cosine scores above zero — BCE does not guarantee this.
            logits = (user_emb @ item_emb.T) / TEMPERATURE  # [B, B]
            labels = torch.arange(b)                         # [B] — class index per row
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
                for user_feat, item_feat in val_loader:
                    b = user_feat.size(0)
                    user_emb = user_tower(user_feat)
                    item_emb = item_tower(item_feat)
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
    with torch.no_grad():
        item_embs = item_tower(torch.tensor(item_features)).numpy().astype(np.float32)

    # Towers already L2-normalize; this is a belt-and-suspenders safeguard.
    faiss.normalize_L2(item_embs)

    index = faiss.IndexFlatIP(EMBED_DIM)   # exact cosine similarity (on unit sphere)
    index.add(item_embs)
    return index


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(user_tower, index, user_features, items, user_archetypes):
    user_tower.eval()

    # Pick one representative user per archetype for the demo
    archetype_labels = ['produce', 'snacks', 'cleaning']
    demo_users = {}
    for label in archetype_labels:
        target_idx = CATEGORIES.index(label)
        matches = np.where(user_archetypes == target_idx)[0]
        if len(matches):
            demo_users[label] = int(matches[0])

    print("\n" + "=" * 72)
    print("  INFERENCE DEMO — Top-10 Recommendations via FAISS cosine similarity")
    print("=" * 72)

    for archetype, user_id in demo_users.items():
        feat = torch.tensor(user_features[user_id]).unsqueeze(0)  # [1, 20]
        with torch.no_grad():
            user_emb = user_tower(feat).numpy().astype(np.float32)  # [1, 64]
        faiss.normalize_L2(user_emb)

        scores, indices = index.search(user_emb, 10)  # [1,10], [1,10]

        # Summarize user profile from preference_scores slice [8:16]
        prefs = user_features[user_id][8:16]
        top_cat_idxs = sorted(range(N_CATS := len(CATEGORIES)), key=lambda i: -prefs[i])[:3]
        cat_summary = "  |  ".join(
            f"{CATEGORIES[c]} ({prefs[c]*100:.0f}%)" for c in top_cat_idxs
        )

        print(f"\n  User {user_id:3d}  [archetype: {archetype}]")
        print(f"  Purchase profile: {cat_summary}")
        print()
        print(f"  {'Rank':<5} {'Item':<28} {'Category':<12} {'Price':<10} {'Cosine':<8}")
        print("  " + "-" * 63)
        for rank, (item_idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            item  = items[item_idx]
            price = PRICE_TIERS[item['price_tier']]
            print(
                f"  {rank:<5} {item['name']:<28} {item['category']:<12} "
                f"{price:<10} {score:.4f}"
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
     Exact search over 200 vectors is fine; at 1M items it is not.

  4. FEATURE STORE
     User features change on every purchase. A feature store (Redis, Feast,
     Tecton) serves pre-computed user vectors at <10 ms latency. Without it,
     you recompute features per request (slow) or serve stale vectors (stale).

  5. HARD NEGATIVE MINING
     In-batch negatives are mostly trivially easy (random unrelated items).
     Hard negatives — items the user didn't buy but that scored highly in a
     prior retrieval pass — dramatically improve recall@K at larger scales.

  6. COLD-START FOR NEW ITEMS
     New items have no interactions. Add a sentence-transformer or bag-of-words
     hash embedding over item text (name, description) as an extra item tower
     input. The model can then embed new items before interactions accumulate.

  7. ONLINE / INCREMENTAL LEARNING
     Retrain the user tower hourly or on purchase triggers. Versioned model
     checkpoints + blue/green FAISS index swaps prevent serving stale embeddings
     that were built against a previous model version.

================================================================================
"""


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("  TWO-TOWER GROCERY RECOMMENDATION PROTOTYPE")
    print("=" * 72)

    # 1. Generate synthetic data
    print("\nGenerating synthetic grocery data...")
    user_features, item_features, items, interactions, user_archetypes = generate_data(
        n_users=500, n_items=200, n_interactions=5000
    )
    print(
        f"  {len(user_features)} users  |  {len(items)} items  |  "
        f"{len(interactions)} interactions"
    )
    print(f"  User feature dim: {user_features.shape[1]}  |  Item feature dim: {item_features.shape[1]}")

    # 2. Train towers
    user_tower, item_tower = train(user_features, item_features, interactions)

    # 3. Build FAISS index
    print("\nBuilding FAISS IndexFlatIP over all item embeddings...")
    index = build_faiss_index(item_tower, item_features)
    print(f"  Index ready: {index.ntotal} vectors, dim={EMBED_DIM}")

    # 4. Retrieve top-10 for demo users
    run_inference(user_tower, index, user_features, items, user_archetypes)

    # 5. Production notes
    print(PRODUCTION_NOTES)
