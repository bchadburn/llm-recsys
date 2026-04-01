# Two-Tower Model — Deep Dive

This document covers the model architecture, feature design, training approach, and the path to a production system. For setup and running instructions see [README.md](README.md).

---

## Architecture overview

A two-tower model (also called a dual encoder) trains two independent neural networks to map different input types into a shared embedding space. Similarity in that space is the retrieval signal.

```
User features (20d)                   Item features (13d)
       │                                      │
  UserTower                              ItemTower
  Linear(20→128)                         Linear(13→64)
  ReLU                                   ReLU
  LayerNorm(128)                         LayerNorm(64)
  Linear(128→64)                         Linear(64→64)
  L2-normalize                           L2-normalize
       │                                      │
  user_emb (64d) ◄────── cosine ──────► item_emb (64d)
```

The towers are entirely separate — they share no weights and never see each other's inputs during training. They are only coupled through the loss function, which rewards high cosine similarity for interacted (user, item) pairs.

---

## Why separate towers?

**Offline pre-computation.** Because the item tower is independent, all 200 item embeddings can be pre-computed once and loaded into FAISS. At query time, only the user tower runs (one forward pass). This makes retrieval O(1) in the number of items.

**Asymmetric inputs.** Users and items are fundamentally different things with different feature types. Forcing them through shared weights would require a fixed-size concatenated input and lose the architectural flexibility to handle each side differently (e.g., the user tower can be deeper or wider than the item tower without affecting the other side).

**The trade-off.** No cross-attention between user and item — the model cannot capture fine-grained interaction effects like "this user hates premium frozen items but loves premium dairy". That expressiveness lives in a downstream re-ranking stage.

---

## Feature schemas

### User features — 20 floats

| Slice | Field | Description |
|---|---|---|
| `[0:8]` | `norm_counts` | log1p purchase count per category, scaled to [0,1] by max |
| `[8:16]` | `prefs` | fraction of purchases per category (sums to 1.0) |
| `[16]` | `total_inter` | log1p(total purchases), normalized |
| `[17]` | `recency` | synthetic recency score in [0,1] |
| `[18]` | `price_sens` | mean price tier of purchases in [0,1] |
| `[19]` | `variety` | Shannon entropy of purchase distribution, normalized to [0,1] |

`norm_counts` and `prefs` are complementary: `norm_counts` carries raw magnitude (heavy buyer vs light buyer), `prefs` carries relative intent (what proportion goes to each category). Together they let the model see both *how much* a user buys and *what* they tend to buy.

`variety` is Shannon entropy: `H = -sum(p_i * log(p_i))`, normalized by `log(8)`. A value near 1 means the user buys evenly across all categories; near 0 means they almost exclusively buy one category.

### Item features — 13 floats

| Slice | Field | Description |
|---|---|---|
| `[0:8]` | `category_onehot` | one-hot over 8 grocery categories |
| `[8:11]` | `price_onehot` | one-hot: [budget, mid, premium] |
| `[11]` | `popularity` | log1p-normalized interaction count in [0,1] |
| `[12]` | `avg_rating` | normalized rating in [0,1] |

One-hot encoding is used for both category and price tier — no ordinal relationship is implied between categories (produce is not "less than" dairy), and the model should not assume mid is halfway between budget and premium in any continuous sense.

---

## Model architecture choices

**User tower is deeper than item tower.** User features are noisier and higher-dimensional — log counts, entropy, and synthetic scalars have more non-linearity to model. Item features are clean categorical signals that need less capacity.

**LayerNorm instead of BatchNorm.** LayerNorm normalizes each sample independently. BatchNorm computes statistics across the batch, which becomes unreliable at inference time when you embed a single user (batch size = 1). LayerNorm works correctly at any batch size.

**L2-normalization on output.** Both towers end with `F.normalize(x, dim=-1)`, projecting outputs onto the unit hypersphere. This makes inner product equivalent to cosine similarity: `cos(u, v) = u · v` when `||u|| = ||v|| = 1`. FAISS `IndexFlatIP` computes inner products, so normalization is the bridge between the model and the index.

**Embedding dimension = 64.** Small enough to keep FAISS memory footprint trivial at 200 items, large enough to represent meaningful structure. Production systems typically use 128–512d.

---

## Training

### Data

5000 (user, item) pairs sampled from the synthetic interaction log. Each user has a dominant category that drives ~63% of their purchases. The bias is strong enough that the model can learn meaningful user-item alignment from 20 epochs.

80/20 train/val split on interaction pairs.

### Loss function

**In-batch negatives with dot-product BCE:**

```python
logits = (user_emb @ item_emb.T) / temperature   # shape [B, B]
labels = torch.eye(B)                             # diagonal = 1.0
loss   = F.binary_cross_entropy_with_logits(logits, labels)
```

For a batch of B=64 pairs:
- The diagonal of `logits` contains the B positive (user, item) scores
- Every off-diagonal cell is treated as a negative — B-1 negatives per example, for free
- BCE trains each cell independently: positive cells toward 1, negative cells toward 0

**Temperature** (0.07) scales the logits before BCE. Lower temperature sharpens the distribution — a small difference in cosine similarity becomes a large difference in loss. 0.07 is the value used in CLIP.

**Why BCE over InfoNCE / cross-entropy?** Both work. Cross-entropy (InfoNCE) treats the problem as "for user i, classify which of B items is the correct one" — a softmax over the row. BCE treats each cell independently. At this small scale with strong category signal, BCE converges quickly. InfoNCE tends to drive positive cosine scores more reliably above zero in practice (see production notes below).

### Optimizer

Adam, lr=1e-3, 20 epochs. No scheduler — the loss flattens quickly because the synthetic category signal is strong and low-dimensional.

---

## FAISS retrieval

**Index type: `IndexFlatIP`**

Exact inner product search over all 200 item embeddings. Since embeddings are L2-normalized, inner product equals cosine similarity.

```python
index = faiss.IndexFlatIP(64)
index.add(item_embeddings)            # [200, 64], float32
scores, indices = index.search(query_emb, k=10)   # query_emb: [1, 64]
```

`IndexFlatIP` does a brute-force scan — no approximation, no training step, deterministic. At 200 items this takes microseconds. It is the right choice here; approximate methods like HNSW or IVF only pay off above ~10k vectors.

---

## Understanding the output scores

Cosine similarity on a unit hypersphere is bounded in [-1, 1]:
- **+1**: vectors point in exactly the same direction (perfect match)
- **0**: vectors are orthogonal (no relationship)
- **-1**: vectors point in opposite directions (maximum mismatch)

In this prototype, scores come out slightly negative (around -0.11 to -0.14). This is a training artifact: the BCE loss only requires positive pairs to score *relatively higher* than negatives within a batch — it does not require the absolute scores to be positive. The model converges to a regime where all scores are small and negative, but the correct items score slightly higher (less negative) than incorrect ones. Retrieval ranking is still correct.

In practice this is addressed by switching to InfoNCE loss (cross-entropy over the full similarity row), which naturally drives positive pair scores above zero because softmax requires the correct item to dominate the entire row, not just be marginally higher.

---

## Path to production

The prototype establishes the core pattern. Here is the evolution path in priority order.

### 1. Switch loss to InfoNCE

Replace BCE with symmetric cross-entropy (NT-Xent / InfoNCE):

```python
labels = torch.arange(B)
loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
```

This is the loss used in CLIP, Google's dual encoder, and most production two-tower systems. It drives positive cosine scores reliably above zero and converges to more interpretable embedding geometries.

### 2. Add ID embeddings

The current user tower sees only hand-crafted features. A production system adds a learned embedding table:

```python
user_id_emb = nn.Embedding(n_users, 32)(user_ids)       # [B, 32]
user_input   = torch.cat([user_id_emb, user_features], dim=-1)  # [B, 52]
```

ID embeddings capture user-specific biases that aggregate features cannot. Same for items. This is the highest-leverage addition after fixing the loss.

### 3. Two-stage retrieval

The two-tower retrieves a candidate set (top-100 to top-500 items). A ranking stage re-scores the candidates using features that are too expensive to compute at full catalog scale:

```
Two-tower (FAISS) → top-100 candidates
        ↓
Cross-encoder or LightGBM ranker → top-10 final results
```

The ranker can attend to joint (user, item) features: interaction history, session context, real-time signals. This recovers the expressiveness that the dual-encoder architecture sacrifices.

### 4. Approximate nearest neighbor at scale

| Item count | Index type | Notes |
|---|---|---|
| < 10k | `IndexFlatIP` | Exact, no setup |
| 10k – 1M | `IndexHNSWFlat` | Sub-ms, >99% recall, graph-based |
| > 1M | `IndexIVFPQ` | Compressed, good for very large catalogs |

At 1M items, brute-force scan over 64-dim float32 vectors would be ~256 MB per query scan — HNSW makes this sub-millisecond with negligible recall loss.

### 5. Feature store

User features change on every purchase. A feature store (Redis, Feast, Tecton) serves pre-computed user feature vectors at <10ms latency. The alternative — recomputing features per request from raw transaction logs — is too slow for real-time serving.

### 6. Hard negative mining

In-batch negatives are mostly trivially easy (a produce buyer is unlikely to interact with cleaning products). Hard negatives — items that score highly for a user but were not purchased — dramatically improve the quality of the learned representations. Mined from a previous retrieval pass and injected into training batches.

### 7. Cold-start

New items have zero interactions. Options:
- Add a text encoder (sentence-transformer) over item name/description to the item tower
- Use content-based fallback (return top items from the item's category)
- Propagate embeddings from similar existing items

### 8. Online learning

Retrain the user tower on an hourly or purchase-triggered cadence. Item tower retrains daily or when new items are added. Requires:
- Versioned model checkpoints
- Blue/green FAISS index swaps (build the new index against the new model, then atomically swap it in)
- Monitoring for embedding drift (cosine similarity between old and new embeddings for the same entities)
