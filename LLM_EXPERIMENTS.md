# LLM Enhancement Experiments — Next Steps

Three self-contained Python scripts to add to the recommendation system, each demonstrating a distinct LLM integration pattern. The current system ends at: two-tower (InfoNCE) → FAISS retrieval → LightGBM lambdarank. None of those stages use natural language at inference time.

---

## Experiment 1: `llm_user_narration.py` — Zero-shot text retrieval

**Pattern:** User features → text profile → sentence-transformer → query existing 384d item index directly (no UserTower training).

**What it shows:** Cold-start retrieval without a trained user tower. A text description of a user lands near items in the same semantic space as item descriptions. Measures how much the trained UserTower adds over zero-shot.

**Two variants:**
- Template narration (no API key needed) — rule-based conversion of numeric features to text
- LLM narration (`--llm` flag) — Claude enriches the template with inferred meal habits, household type, lifestyle signals

**Key output:** Side-by-side retrieval results for template vs LLM profile vs trained two-tower. Category precision (% of top-10 in dominant category) as a quick quality metric.

**Run (template only):**
```bash
uv run --with sentence-transformers --with faiss-cpu --with torch --with numpy \
    --with python-dotenv python llm_user_narration.py
```

**Run (with LLM enrichment):**
```bash
uv run --with anthropic --with sentence-transformers --with faiss-cpu --with torch \
    --with numpy --with python-dotenv python llm_user_narration.py --llm
```

---

## Experiment 2: `llm_reranker.py` — Contextual LLM reranking

**Pattern:** FAISS top-20 candidates + natural-language shopping context → Claude → reranked list with reasoning.

**What it shows:** Dot-product retrieval is blind to session context. An LLM can use signals that embeddings fundamentally cannot express — occasion, dietary goals, cart contents. Same user, different context → meaningfully different ranking.

**Contexts to demonstrate:**
- Baseline (no context — pure embedding-based)
- `"Preparing a Mediterranean dinner for two tonight"`
- `"On a high-protein diet, focusing on muscle building"`
- `"Quick road trip snacks, nothing that needs refrigeration"`

**Key output:** Three-column comparison — FAISS order | LightGBM order | LLM-reranked — for each context, with Claude's top-3 reasoning visible.

**Design note:** LLM reranker is a final-stage filter over a small candidate set (10–20 items). Not appropriate at retrieval scale; cost and latency only work when the set is already small.

**Run:**
```bash
uv run --with anthropic --with sentence-transformers --with faiss-cpu --with torch \
    --with numpy --with lightgbm --with python-dotenv python llm_reranker.py
```

---

## Experiment 3: `llm_item_enrichment.py` — Offline LLM-generated item descriptions

**Pattern:** Replace template item descriptions with Claude-generated rich descriptions → re-embed → measure Recall@K improvement.

**Current template:** `"Organic Spinach - produce - budget quality"`

**LLM-enriched target:**
> "Organic spinach is a nutrient-dense leafy green popular with health-conscious shoppers and home cooks. Frequently used in salads, smoothies, pasta, and stir-fries. Often purchased alongside garlic, tomatoes, and other fresh produce."

**What it shows:** Richer content → richer embeddings → better zero-shot retrieval. Quantified by Recall@K and NDCG@K on held-out interactions, using the same template user narration for both indices so the only variable is item embedding quality.

**Implementation notes:**
- ~200 Claude API calls (one per item), cached to `item_descriptions_cache.json`
- Subsequent runs use the cache — no API key needed after first run
- Evaluation reuses the `eval.py` framework but adapts it for the text-index path

**Run (first time — generates + caches):**
```bash
uv run --with anthropic --with sentence-transformers --with faiss-cpu --with torch \
    --with numpy --with python-dotenv python llm_item_enrichment.py
```

**Run (after caching):**
```bash
uv run --with sentence-transformers --with faiss-cpu --with torch \
    --with numpy --with python-dotenv python llm_item_enrichment.py
```

---

## Setup required

Add `ANTHROPIC_API_KEY` to `~/.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

Add to `requirements.txt`:
```
anthropic>=0.25.0
python-dotenv>=1.0.0
```

Use `claude-haiku-4-5-20251001` for all API calls — cheap and fast for short completions. Experiment 3 costs ~$0.01 total for all 200 item descriptions.

---

## Natural extension (Experiment 4): Offline relationship mining / LLM distillation

Not implemented but worth considering: use Claude to rate user-item affinity for a sample of (user-archetype, item) pairs, then use these ratings as soft training labels for the two-tower model. This is knowledge distillation — the LLM's understanding of grocery shopping encoded into the model's weights rather than called at inference time. Would improve the trained model without adding inference latency.
