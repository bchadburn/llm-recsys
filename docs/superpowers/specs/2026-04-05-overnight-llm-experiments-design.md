# Overnight LLM Experiments — Design Spec
**Date:** 2026-04-05  
**Status:** Approved

## Overview

Four sequential LLM experiments running overnight on the Instacart recommendation system. Each experiment is self-contained, builds on the existing two-tower + FAISS + LightGBM/XGBoost pipeline, and targets a distinct LLM integration pattern. Results are consolidated into a single `overnight_report.md` for morning review.

The orchestration script (`run_overnight.sh`) runs all four experiments in sequence, captures stdout/stderr per experiment, and writes the final report.

---

## Experiment 1: LLM Item Enrichment

**File:** `llm_item_enrichment.py`  
**Pattern:** Offline LLM-generated content → richer embeddings → better zero-shot retrieval  
**Model:** `claude-haiku-4-5-20251001`

### What it does
Replaces the template item description (`"Organic Spinach - produce - budget quality"`) with a Claude-generated rich description that includes use cases, co-purchase context, meal associations, and shopper type. Re-embeds all ~5k items with sentence-transformer and builds a new FAISS index. Evaluates Recall@K and NDCG@K using the same `eval.py` framework, comparing enriched vs template index.

### Implementation details
- ~5k API calls to Haiku, cached to `item_descriptions_cache.json` after first run
- Subsequent runs skip API calls and use cache — safe to re-run
- Estimated cost: ~$0.10–0.20 for all items
- Reuses `eval.py` evaluate() function with the text-embedding index path
- Output: Recall@5/10/20 and NDCG@5/10/20 for template vs LLM-enriched

### Success criteria
Recall@20 improves over template baseline (currently ~0.010).

---

## Experiment 2: Zero-shot User Narration

**File:** `llm_user_narration.py`  
**Pattern:** User features → text profile → query item index without a trained user tower  
**Model:** `claude-haiku-4-5-20251001`

### What it does
Converts user feature vectors into natural language profiles two ways:
1. **Template:** Rule-based conversion (e.g. "Heavy produce buyer (39%), moderate dairy (22%), light frozen (7%)")
2. **LLM-enriched:** Claude infers meal habits, household type, lifestyle signals from the template

Both profiles are embedded with the same sentence-transformer as items, then queried directly against the enriched FAISS index from Exp 1. Results compared against the trained two-tower.

### Implementation details
- Runs on 10 representative users (2 per archetype: produce, dairy, frozen, snacks, household)
- LLM calls per user: 1 (profile enrichment) — ~10 total, negligible cost
- Output: side-by-side top-10 for template narration vs LLM narration vs trained two-tower, plus category precision (% of top-10 in dominant category) as quick quality signal

### Success criteria
Demonstrates measurable gap between zero-shot and trained retrieval; LLM narration improves on template.

---

## Experiment 3: Contextual LLM Reranker

**File:** `llm_reranker.py`  
**Pattern:** FAISS candidates + session context → LLM reranking with reasoning  
**Model:** `claude-haiku-4-5-20251001`

### What it does
For a sample of 5 users, retrieves FAISS top-20 using the trained two-tower, then asks Claude to rerank the candidates given four different shopping contexts:
1. No context (baseline)
2. `"Preparing a Mediterranean dinner for two tonight"`
3. `"On a high-protein diet, focusing on muscle building"`
4. `"Quick road trip snacks, nothing that needs refrigeration"`

Outputs a three-column comparison per context: FAISS order | LightGBM order | LLM-reranked, with Claude's top-3 reasoning visible.

### Implementation details
- ~20 API calls total (5 users × 4 contexts)
- Depends on trained two-tower and LightGBM model from `main.py` — loads existing artifacts or re-trains if not found
- Output is qualitative (no Recall@K) — demonstrates contextual sensitivity that embeddings can't express

### Success criteria
LLM reranking produces meaningfully different lists across contexts for the same user.

---

## Experiment 4: Synthetic Context Injection

**File:** `synthetic_context.py`  
**Pattern:** Augment training data with synthetic session occasions → context-aware two-tower  

### What it does
Assigns each training interaction a randomly sampled synthetic occasion from a fixed set:
- `weeknight_dinner`, `meal_prep`, `party_hosting`, `health_diet`, `quick_snack`, `road_trip`

Occasion is one-hot encoded and appended to user feature vectors, expanding user feature dim from 49 to 55. Retrains the two-tower with context-aware user features. Evaluates Recall@K vs context-free baseline.

### Implementation details
- No LLM calls — fully synthetic, no API cost
- Occasions assigned pseudo-randomly per interaction using SEED=42 for reproducibility
- At eval time, a random occasion is sampled per user query
- Same training config as main.py (20 epochs, InfoNCE, FAISS)
- Output: Recall@5/10/20 and NDCG@5/10/20 vs context-free baseline

### Success criteria
Establishes whether synthetic context signal improves or hurts retrieval — hypothesis is neutral/slight improvement since occasions are random and don't correlate with actual purchase behavior.

---

## Orchestration

**File:** `run_overnight.sh`

```
1. Verify ANTHROPIC_API_KEY is set
2. Run llm_item_enrichment.py       → logs/exp1_enrichment.log
3. Run llm_user_narration.py        → logs/exp2_narration.log
4. Run llm_reranker.py              → logs/exp3_reranker.log
5. Run synthetic_context.py         → logs/exp4_synthetic.log
6. Run generate_report.py           → overnight_report.md
```

Each experiment logs independently. If one fails, the script continues to the next. `generate_report.py` reads all four logs and compiles a structured markdown summary.

---

## Report Structure (`overnight_report.md`)

```
# Overnight LLM Experiments — Results
Date: <timestamp>

## Summary Table
| Experiment | Key Metric | Result | vs Baseline |

## Exp 1: Item Enrichment
[Recall/NDCG table, template vs LLM-enriched]

## Exp 2: Zero-shot Narration  
[Top-10 comparison for 2 representative users]

## Exp 3: Contextual Reranker
[Side-by-side for 1 user × 4 contexts, with reasoning]

## Exp 4: Synthetic Context
[Recall/NDCG table, context-aware vs baseline]

## Observations & Next Steps
[Auto-generated notes on what improved, what didn't, suggested follow-ups]
```

---

## Dependencies

Add to `requirements.txt`:
```
anthropic>=0.25.0
python-dotenv>=1.0.0
```

API key in `~/.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

Estimated total API cost: ~$0.30–0.50 for all experiments combined.
