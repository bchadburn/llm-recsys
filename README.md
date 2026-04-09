# Grocery Recommendation System — LLM Integration Research

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

## Key findings

**Exp 1 — Item enrichment:** LLM descriptions are 6.3× richer than template text
(65 vs 10 words avg). Department coverage 74.6%. Marginal Recall improvement over
template embeddings — semantic signal exists but the sentence-transformer already
captures most of it from item names alone.

**Exp 3 — LLM reranker:** Context-aware reranking demonstrably shifts rankings
(same user, different context → different top items). Aggregate Recall is competitive
with LightGBM at a much higher API cost — value is in the qualitative context-
sensitivity, not raw recall.

**Exp 4 — Synthetic context:** Negative result. Random occasion labels add noise;
no recall improvement. Confirms ground-truth occasion signal would be needed.

**Exp 5 — Dual-head tower:** Gate weight α converges to 0.986 (frozen semantic)
and 0.889 (trainable) — optimizer heavily prefers the semantic head. Trainable
variant achieves lowest val loss (2.7699 vs 2.8574 baseline), suggesting the
linear semantic projection learns faster than the collaborative MLP.

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
