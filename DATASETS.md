# Dataset Options for Recommendation System Experiments

## TL;DR Recommendation

| Goal | Best dataset |
|---|---|
| Drop-in replacement for synthetic grocery data | **Instacart** |
| LLM item enrichment / text embedding experiments | **Amazon Grocery subset** |
| Implicit feedback / multi-event modeling | **RetailRocket** |
| Stay self-contained, no download required | **Improve synthetic data** (see below) |

---

## Option 1: Instacart Market Basket Analysis ⭐ Best fit for this project

**Why:** Real grocery data. Product names, aisle taxonomy, departments — directly replaces the synthetic catalog. Has reorder signals and temporal features the current synthetic data lacks entirely.

**What's in it:**
- 3M+ orders from 200k+ users across ~50k products
- `orders.csv` — user_id, order sequence, day_of_week, hour_of_day, days_since_prior_order
- `products.csv` — product_id, product_name, aisle_id, department_id
- `aisles.csv` / `departments.csv` — 134 aisles, 21 departments (the real category taxonomy)
- `order_products__prior.csv` — (order_id, product_id, add_to_cart_order, reordered)

**What this unlocks vs synthetic data:**
- Real product names → meaningful text embeddings (LLM enrichment experiments)
- `reordered` flag → distinguish routine re-purchases from new discoveries (better labels)
- `days_since_prior_order` → real recency signal for user features
- `add_to_cart_order` → implicit preference ranking within a basket
- 134 aisles vs 8 categories → much richer item taxonomy

**Download:**
```bash
# Requires free Kaggle account + kaggle CLI
pip install kaggle
kaggle competitions download -c instacart-market-basket-analysis
```
Or download manually: https://www.kaggle.com/competitions/instacart-market-basket-analysis/data

**License:** Non-commercial use only (fine for learning/research).

**Effort to integrate:** Medium. `data.py` needs a new `load_instacart()` function that maps the 6 CSVs to the same `(user_features, item_features, items, interactions)` format `main.py` expects. The key mapping work is building user feature vectors from order history (purchase counts per department, recency, reorder rate) and item features from product name + aisle + department.

---

## Option 2: Amazon Reviews 2023 — Grocery and Gourmet Food ⭐ Best for LLM experiments

**Why:** Every item has a real product description, price, and category. This is ideal for the `llm_item_enrichment.py` experiment — you can compare sentence-transformer embeddings of the original Amazon description vs an LLM-rewritten one. Also has review text, which can feed user preference signals.

**What's in it (Grocery and Gourmet Food subset):**
- Product metadata: title, description, features list, price, category path, brand
- User reviews: rating (1–5), review text, verified purchase flag, timestamp
- User-item interactions with fine-grained timestamps

**Download (via Hugging Face `datasets`):**
```python
from datasets import load_dataset

# Item metadata
meta = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_meta_Grocery_and_Gourmet_Food",
    split="full", trust_remote_code=True
)

# User reviews / interactions
reviews = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_Grocery_and_Gourmet_Food",
    split="full", trust_remote_code=True
)
```

**License:** Research use. Source: https://amazon-reviews-2023.github.io/

**Effort to integrate:** Low for item enrichment experiments (just swap item descriptions). Higher if building full interaction data — ratings need to be converted to implicit feedback (treat ratings ≥ 4 as positive interactions).

---

## Option 3: RetailRocket E-commerce Dataset

**Why:** Real implicit feedback with three event types — view, add-to-cart, transaction. Good for teaching about weighting different event signals (a transaction is stronger than a view; the current system treats all interactions as equal).

**What's in it:**
- 2.75M events over 4.5 months, 1.4M users
- `events.csv`: timestamp, visitorid, event (view/addtocart/transaction), itemid
- Item properties exist but are hashed — **no readable product names or categories**

**The catch:** Because all item IDs and properties are hashed, there's no text to embed. This makes it incompatible with LLM text experiments (query expansion, item enrichment). Useful for implicit feedback modeling but not the LLM experiments planned here.

**Download:** https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset (free, no competition required)

---

## Option 4: OTTO Multi-Objective Recommender (session-based)

12.9M sessions, 220M events, 1.8M items. Designed for session-based / next-item recommendation. Like RetailRocket, all IDs are anonymized — no item content. Overkill scale for learning experiments and doesn't fit the LLM text experiments. Skip unless specifically exploring session modeling at scale.

---

## Option 5: Improve the Synthetic Data (no download needed)

If external data isn't worth the setup overhead, the synthetic data in `data.py` can be made significantly more realistic without changing the downstream interface:

**High-value additions:**

1. **Reorder behavior** — track which items users re-buy vs. discover new. Reorder rate is a strong signal (routine staples vs. impulse/exploration purchases).

2. **Temporal patterns** — simulate week/time-of-day purchase bias (dairy on weekday mornings, snacks Friday evenings). Adds a recency dimension to user features.

3. **Basket co-occurrence** — items bought together in the same session (chicken + garlic + pasta). This creates a richer interaction graph and opens up graph-based methods.

4. **Richer item descriptions** — add a `description` field per item (currently items only have name + category + price tier). Used by `get_item_text_embeddings()` to generate more informative embeddings.

5. **Implicit feedback signal strength** — currently every (user, item) interaction is treated as equal. Add purchase count per (user, item) pair and use it as interaction weight in training (upweight items bought 5+ times).

6. **Cold-start items** — add a small set of brand-new items with zero interactions to test cold-start handling directly.

---

## Recommended path

1. **Start with Instacart** if you want real data without switching domains. The aisle taxonomy and product names make the LLM experiments much more interesting (real item names produce meaningfully different embeddings than synthetic ones).

2. **Add Amazon Grocery metadata** as a secondary source for item descriptions — cross-reference by product name to enrich Instacart items with Amazon's description text.

3. **Improve synthetic data** as a parallel track — adding reorder behavior and temporal patterns costs one afternoon and makes the existing evaluation metrics more meaningful immediately.
