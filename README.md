# Grocery Recommendation Prototype

A minimal two-tower retrieval system built in PyTorch. A user encoder and an item encoder are trained independently, their outputs placed in the same embedding space, and FAISS is used to retrieve the nearest items to a query user at inference.

See [TWO_TOWER_MODEL.md](TWO_TOWER_MODEL.md) for a deep dive on the model architecture, feature schemas, training loss, and production path.

---

## What was built

```
recommendation_system/
├── data.py       synthetic data generation + PyTorch Dataset
├── model.py      UserTower and ItemTower (PyTorch nn.Module)
├── main.py       training loop, FAISS index, inference demo, evaluation
├── ranker.py     LightGBM lambdarank re-ranking stage (two-stage retrieval)
├── eval.py       offline evaluation: Recall@K and NDCG@K
└── requirements.txt
```

**No external dataset needed.** All grocery data is generated in `data.py`: 500 users, 200 items across 8 categories, 5000 purchase interactions.

---

## How to run

```bash
uv run --with torch --with faiss-cpu --with numpy python main.py
```

Or install first and then run directly:

```bash
pip install -r requirements.txt
python main.py
```

Expected output:
1. Training loss every 5 epochs for both approaches (one-hot features, text embeddings)
2. Side-by-side top-10 comparison: one-hot vs text-embedding retrieval, three user archetypes
3. Offline evaluation table — Recall@5/10/20 and NDCG@5/10/20 for both approaches
4. LightGBM ranker training log, then FAISS top-10 vs re-ranked top-10 side by side
5. Production notes

---

## System design

```
┌─────────────────────────────────────────────────────────┐
│                     TRAINING                            │
│                                                         │
│  user features (20d) ──► UserTower ──► user_emb (64d)  │
│  item features (13d) ──► ItemTower ──► item_emb (64d)  │
│                                                         │
│  loss = InfoNCE( user_emb @ item_emb.T / temp )        │
│         ↑ diagonal = positive pairs                     │
│         ↑ off-diagonal = in-batch negatives             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     INFERENCE                           │
│                                                         │
│  All 200 items ──► ItemTower ──► FAISS index (64d)     │
│                                                         │
│  Query user ──► UserTower ──► query_emb (64d)          │
│                    │                                    │
│                    └──► FAISS.search(k=10)              │
│                              │                          │
│                              └──► top-10 items          │
└─────────────────────────────────────────────────────────┘
```

The key design constraint: both towers output **L2-normalized, 64-dim embeddings**. This means inner product equals cosine similarity, and FAISS `IndexFlatIP` (inner product) gives exact cosine similarity without any post-processing.

---

## The two-tower approach

The central idea is to keep the user and item encoders completely separate. They never see each other's inputs — they only interact through the shared embedding space they are trained to agree on.

**Why this matters at scale:** both towers can be run offline. All item embeddings are pre-computed and indexed once. At request time you only run the user tower (one forward pass on 20 numbers), then do a fast vector search. This is O(1) in the number of items.

**The trade-off:** because the towers never jointly attend to (user, item) pairs, they cannot model fine-grained interaction effects. That is the job of a ranking stage that sits on top (see production notes in `main.py`).

---

## Data

All data is generated in `data.py` at runtime — just run the script.

**8 grocery categories:** produce, dairy, meat, bakery, beverages, snacks, frozen, cleaning

**Items (200 total, 25 per category):** each item has a category, price tier (budget/mid/premium), popularity score, and average rating.

**Users (500):** each user has a dominant category (their "archetype") that drives ~63% of their purchases. The remaining ~37% are spread across other categories. This creates realistic, separable user profiles.

**Interactions (5000):** sampled per user with category bias. Used as positive (user, item) pairs during training. No explicit negatives are constructed — the training loss mines negatives from within each batch.

---

## Ranking stage

`ranker.py` implements the second stage of a two-stage retrieval system. After FAISS retrieves the top-20 candidates per user, a LightGBM model with the lambdarank objective re-scores them using features that the dot-product two-tower cannot express:

| Feature | Why it can't live in the two-tower |
|---|---|
| `user_emb ⊕ item_emb` (128d) | Two-tower collapses this to a scalar dot product |
| FAISS dot-product score | Retrieval signal, passed through as a feature |
| Item popularity | Global prior; entangled with category signal in the tower |
| Item price tier | Categorical; GBDT splits handle this cleanly |
| `user_price_sensitivity × price_tier` | Explicit cross-feature interaction — the two towers encode each side independently and cannot represent this joint signal |

Relevance labels are synthetic: items the user purchased in the training interactions score 1, all other FAISS candidates score 0. The ranker trains on 200 users × 20 candidates = 4000 (user, item) pairs.

---

## Offline evaluation

`eval.py` measures retrieval quality against the same 20% held-out validation split used during model training (reproduced deterministically from `SEED=42`).

**Recall@K** — the retrieval metric. Of all items a user actually purchased, what fraction appear in the top-K results? Low Recall@20 means the ranker is working with an impoverished candidate set; items not retrieved by FAISS are invisible to every later stage.

**NDCG@K** — the ranking metric. Normalized Discounted Cumulative Gain penalises relevant items that appear lower in the list. NDCG distinguishes a model that places the best match at rank 1 from one that buries it at rank 10 — Recall alone cannot.

Both metrics are reported at K=5, 10, 20 for both approaches (one-hot features and text embeddings).

---

## Key design decisions

**InfoNCE loss (symmetric cross-entropy) over triplet or BCE.** For each user in a batch, the loss treats the remaining B-1 items as negatives via softmax — forcing the positive item to win a competition, not just clear an independent threshold. This drives positive cosine scores above zero naturally. See [TWO_TOWER_MODEL.md](TWO_TOWER_MODEL.md) for detail on why BCE fails to do this.

**LayerNorm over BatchNorm.** Inference runs one user at a time (batch size 1). BatchNorm statistics break at batch size 1; LayerNorm is instance-wise and safe.

**`IndexFlatIP` (exact search).** 200 items is far below the ~10k threshold where approximate search pays off. Exact search over 200 vectors takes microseconds. No index training step required.

**faiss-cpu** is used for vector search. It is fully open source (Meta AI, MIT license) — no cost.

---

## Output interpretation

Cosine scores are positive (typically 0.4–0.5 in this prototype) because InfoNCE loss forces the positive item to win a full softmax competition — not just clear an independent threshold. Scores closer to 1.0 indicate stronger alignment between the user embedding and item embedding. The ranking within each user's results is the signal that matters for retrieval.
