# Grocery Recommendation System — LLM Integration Research

Two-tower retrieval (PyTorch + FAISS) with LightGBM re-ranking, plus five experiments
testing where LLM integration actually helps — and where it doesn't.

Dataset: [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis)
(~2,000 users, ~5,000 items, ~1.5M interactions). Synthetic data used by default.

---

## Quickstart

```bash
pip install -r requirements.txt

# Synthetic data (no download needed)
python main.py

# Instacart data (download from Kaggle first — see DATASETS.md)
python main.py --data-dir data/instacart/
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  RETRIEVAL (two-tower, InfoNCE loss)                            │
│  user features ──► UserTower ──► user_emb (64d) ──┐            │
│  item features ──► ItemTower ──► item_emb (64d) ──┴──► FAISS   │
└─────────────────────────────────────────────────────────────────┘
          │ top-20 candidates
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  RE-RANKING                                                     │
│  · LightGBM (lambdarank) — joint user×item features            │
│  · LLM reranker — natural language context awareness           │
└─────────────────────────────────────────────────────────────────┘
```

See [TWO_TOWER_MODEL.md](TWO_TOWER_MODEL.md) for architecture details and production notes.

---

## Results

Two-tower baseline (Instacart, 2,000 users, 80/20 split):

| Metric | Value |
|--------|-------|
| Recall@5 | 0.0592 |
| Recall@10 | 0.0904 |
| Recall@20 | 0.1298 |
| NDCG@10 | 0.5005 |

LLM experiment outcomes:

| Experiment | Result |
|-----------|--------|
| **Exp 1 — Item enrichment** | Marginal gain. LLM descriptions 6.3× richer but sentence-transformer already captures most signal from item names alone |
| **Exp 2 — Zero-shot narration** | Negative. Category precision@10: template=0.767, LLM=0.167 — text profile can't replace a trained tower |
| **Exp 3 — LLM reranker** | Context-aware reranking demonstrably shifts rankings (same user, different context → different top items). Recall competitive with LightGBM at much higher API cost |
| **Exp 4 — Synthetic context** | Negative. Synthetic occasion labels add noise; R@20 drops 0.1298 → 0.1273 |
| **Exp 5 — Dual-head tower** | Gate α → 0.986 (frozen) / 0.889 (trainable) — optimizer heavily prefers semantic head. Trainable variant achieves best val loss (2.7699 vs 2.8574 baseline) |

**Takeaway:** LLMs add value for context-aware reranking, not for replacing learned retrieval representations.

---

## Files

| File | Purpose |
|------|---------|
| `main.py` | Training pipeline (two-tower + FAISS + LightGBM) |
| `model.py` | UserTower, ItemTower, DualHeadItemTower |
| `ranker.py` | LightGBM lambdarank re-ranking stage |
| `eval.py` | Recall@K / NDCG@K evaluation |
| `data.py` / `data_instacart.py` | Synthetic / Instacart data loaders |
| `xgb_model.py` | XGBoost direct ranker (Instacart only, Approach C) |
| `llm_item_enrichment.py` | Exp 1 — Claude item descriptions → embeddings |
| `llm_user_narration.py` | Exp 2 — Zero-shot user profile retrieval |
| `llm_reranker.py` | Exp 3 — Context-aware LLM reranking |
| `synthetic_context.py` | Exp 4 — Occasion injection + retrain |
| `exp5_dual_head.py` | Exp 5 — Dual-head item tower with learned gate |
| `generate_report.py` | Parse experiment logs → `overnight_report.md` |
| `api/` | FastAPI inference service (see below) |
| `evals/` | Eval pipeline for descriptions and reranker quality |

---

## Running the API

FastAPI service wrapping two-tower + LightGBM inference. Trains on synthetic data at startup (~30s).

```bash
# Local
pip install -r requirements-api.txt
uvicorn api.main:app --reload

# Docker
docker compose up
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/recommend` | Top-K recommendations for a user |

```bash
curl -s -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 0, "top_k": 5}'
```

Interactive docs: http://localhost:8000/docs

---

## Eval pipeline

```bash
# Description quality (no API calls)
python evals/run_evals.py --descriptions

# Reranker eval (~40 LLM API calls)
python evals/run_evals.py --reranker --reranker-users 10

# Compare runs
python evals/run_evals.py --compare
```

---

## Overnight run

```bash
bash run_overnight.sh > logs/orchestration.log 2>&1 &
python generate_report.py  # produces overnight_report.md from logs
```

---

## Stack

- **PyTorch** — two-tower training (GPU when available)
- **FAISS** `IndexFlatIP` — exact inner-product vector search
- **LightGBM** — lambdarank re-ranking
- **FastAPI + uvicorn** — inference API
- **sentence-transformers** `all-MiniLM-L6-v2` — item text embeddings
- **Claude Haiku** `claude-haiku-4-5-20251001` — descriptions, profiles, reranking
