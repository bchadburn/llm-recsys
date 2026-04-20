# Grocery Recommendation System — LLM Integration Research

![CI](https://github.com/bchadburn/llm-recsys/actions/workflows/ci.yml/badge.svg)

A research codebase exploring LLM integration patterns for grocery recommendation
on the [Instacart dataset](https://www.kaggle.com/c/instacart-market-basket-analysis)
(~2,000 users, ~5,000 items, ~1.5M interactions).

The core is a two-tower retrieval model (PyTorch + FAISS) with a LightGBM re-ranking
stage. Five experiments layer in LLM capabilities on top of that foundation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  RETRIEVAL (two-tower, InfoNCE loss)                            │
│                                                                 │
│  user features ──► UserTower ──► user_emb (64d) ──┐           │
│  item features ──► ItemTower ──► item_emb (64d) ──┤           │
│                                                    └► FAISS     │
└─────────────────────────────────────────────────────────────────┘
          │ top-20 candidates
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  RE-RANKING                                                     │
│  · LightGBM (lambdarank) — joint user×item features            │
│  · LLM reranker — natural language context awareness           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Experiments

| Script | Pattern | Key question |
|--------|---------|-------------|
| `llm_item_enrichment.py` | Offline LLM content → richer embeddings | Do Claude item descriptions improve FAISS retrieval over template text? |
| `llm_user_narration.py` | Zero-shot user profile → item retrieval | Can a natural language user profile replace a trained user tower? |
| `llm_reranker.py` | FAISS candidates + context → Claude reranking | Does LLM reranking beat LightGBM? Does context shift rankings correctly? |
| `synthetic_context.py` | Synthetic occasion injection → retrain | Does adding occasion labels to features improve two-tower performance? |
| `exp5_dual_head.py` | Dual-head item tower with learned gate (α) | Are semantic embeddings destroyed by MLP training? Can a frozen head recover them? |

---

## Results

Two-tower baseline (trained, 2,000 users, 80/20 split):

| Metric | Value |
|---|---|
| Recall@5 | 0.0592 |
| Recall@10 | 0.0904 |
| Recall@20 | 0.1298 |
| NDCG@10 | 0.5005 |

LLM experiment outcomes vs baseline:

| Experiment | Key result |
|---|---|
| **Exp 1 — Item enrichment** | LLM descriptions 6.3× richer (65 vs 10 words). Marginal Recall gain (R@10: 0.0019 vs 0.0010 zero-shot) — sentence-transformer already captures most signal from item names alone |
| **Exp 2 — Zero-shot narration** | Negative. Category precision@10: template=0.767, LLM=0.167. Zero-shot text profile can't replace a trained tower |
| **Exp 3 — LLM reranker** | Context-aware reranking demonstrably shifts rankings (same user, different context → different top items). Aggregate Recall competitive with LightGBM at much higher API cost — value is qualitative context-sensitivity |
| **Exp 4 — Synthetic context** | Negative. Synthetic occasion labels add noise; R@20 drops from 0.1298 → 0.1273. Ground-truth occasion signal would be needed |
| **Exp 5 — Dual-head tower** | Gate weight α → 0.986 (frozen semantic) / 0.889 (trainable) — optimizer heavily prefers semantic head. Trainable variant achieves lowest val loss (2.7699 vs 2.8574 baseline) |

Honest takeaway: off-the-shelf LLM integration patterns don't reliably improve retrieval metrics on this dataset. The experiments expose *where* LLMs add value (context-aware reranking) vs. where they don't (replacing learned representations).

---

## Eval pipeline

```
evals/
├── prompt_registry.py      Hash + version-track every prompt template
├── description_evals.py    Model-free quality metrics for LLM descriptions
├── reranker_evals.py       Reasoning alignment, hallucination rate, context sensitivity
├── run_evals.py            CLI orchestrator — saves timestamped JSON results
└── results/                Eval runs (gitignored)
```

```bash
# Run description quality eval (no API calls — reads cached descriptions)
uv run --with python-dotenv --with sentence-transformers --with faiss-cpu \
    --with lightgbm --with xgboost --with torch \
    python evals/run_evals.py --descriptions

# Run reranker eval on 10 users (uses LLM API, ~40 calls)
uv run --with anthropic --with python-dotenv --with faiss-cpu \
    --with lightgbm --with xgboost --with torch \
    python evals/run_evals.py --reranker --reranker-users 10

# Compare results across runs
python evals/run_evals.py --compare
```

---

## Running experiments

```bash
# Full overnight run
bash run_overnight.sh > logs/orchestration.log 2>&1 &

# Single experiment (no-LLM mode where supported)
uv run --with anthropic --with sentence-transformers --with faiss-cpu \
    --with python-dotenv --with lightgbm --with xgboost --with torch \
    python llm_item_enrichment.py --no-llm
```

---

## Offline evaluation metrics

**Recall@K** — of all items a user purchased (held-out 20%), what fraction appear
in the top-K results? Low Recall@20 means the ranker works from an impoverished
candidate set; items FAISS misses are invisible to every later stage.

**NDCG@K** — normalized discounted cumulative gain. Penalises relevant items that
appear lower in the ranked list. Distinguishes placing the best match at rank 1
vs. rank 10; Recall alone cannot.

Both metrics use the same deterministic 80/20 interaction-level split (`SEED=42`)
reproduced from training.

---

## Stack

- **PyTorch** — two-tower model training (GPU when available)
- **FAISS** — exact inner-product vector search (`IndexFlatIP`)
- **LightGBM** — lambdarank re-ranking stage
- **sentence-transformers** (`all-MiniLM-L6-v2`) — item text embeddings
- **Claude Haiku** (`claude-haiku-4-5-20251001`) — descriptions, profiles, reranking
- **uv** — dependency management; no virtualenv setup required
