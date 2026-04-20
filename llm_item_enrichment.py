"""
Experiment 1: LLM Item Enrichment
Pattern: Offline LLM-generated content → richer embeddings → better retrieval

Generates rich Claude descriptions for all Instacart items, re-embeds with
sentence-transformer, and measures Recall@K / NDCG@K improvement over the
template baseline (plain "product_name - aisle - dept" descriptions).

Caches API responses to item_descriptions_cache.json so re-runs are free.
"""

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path.home() / '.env')

DATA_DIR    = Path('data/instacart')
CACHE_FILE  = Path('item_descriptions_cache.json')
MODEL       = 'claude-haiku-4-5-20251001'
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE  = 256
API_DELAY   = 0.05  # seconds between API calls


def load_data():
    from data_instacart import load_instacart
    print("Loading Instacart data...")
    user_features, item_features, items, interactions, user_archetypes, up_stats = load_instacart(
        str(DATA_DIR), n_users=2000, n_items=5000, n_interactions=1500000,
    )
    print(f"  {len(items)} items | {len(interactions)} interactions")
    return user_features, item_features, items, interactions


def make_template_description(item: dict) -> str:
    return f"{item['name']} - {item.get('aisle', 'unknown aisle')} - {item.get('department', 'unknown dept')}"


def make_llm_description(client, item: dict) -> str:
    prompt = (
        f"Write a 2-sentence product description for a grocery item that would help a "
        f"recommendation system understand what kind of shopper buys it and what meal occasions "
        f"it fits. Be specific about use cases and co-purchase context.\n\n"
        f"Product: {item['name']}\n"
        f"Aisle: {item.get('aisle', 'unknown')}\n"
        f"Department: {item.get('department', 'unknown')}\n\n"
        f"Description:"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=120,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return response.content[0].text.strip()


def generate_descriptions(items: list, use_llm: bool = True) -> dict:
    cache = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())
        print(f"  Loaded {len(cache)} cached descriptions")

    if not use_llm:
        return {str(i): make_template_description(item) for i, item in enumerate(items)}

    import anthropic
    client = anthropic.Anthropic()

    uncached = [(i, item) for i, item in enumerate(items) if str(i) not in cache]
    print(f"  Generating LLM descriptions for {len(uncached)} items (cached: {len(cache)})...")

    for idx, (i, item) in enumerate(uncached):
        try:
            cache[str(i)] = make_llm_description(client, item)
        except anthropic.APIError as e:
            print(f"  Warning: API call failed for item {i}: {e}. Using template.")
            cache[str(i)] = make_template_description(item)
        if idx > 0 and idx % 100 == 0:
            print(f"    {idx}/{len(uncached)} done...")
            CACHE_FILE.write_text(json.dumps(cache, indent=2))
        time.sleep(API_DELAY)

    CACHE_FILE.write_text(json.dumps(cache, indent=2))
    print(f"  Saved {len(cache)} descriptions to {CACHE_FILE}")
    return cache


def embed_descriptions(descriptions: dict, items: list, model) -> np.ndarray:
    texts = [descriptions.get(str(i), make_template_description(item)) for i, item in enumerate(items)]
    print(f"  Embedding {len(texts)} descriptions...")
    embs = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
    return embs.astype(np.float32)


def build_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    normed = embs.copy()
    faiss.normalize_L2(normed)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(normed)
    return index


def zero_shot_evaluate(index: faiss.IndexFlatIP, user_features: np.ndarray,
                        interactions: list, model, ks=(5, 10, 20)) -> dict:
    from data_instacart import DEPARTMENTS

    dept_names = DEPARTMENTS
    n_depts = len(dept_names)

    # 80/20 random split (same seed as eval.py)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(interactions))
    val_ints = [interactions[i] for i in perm[int(0.8 * len(interactions)):]]
    val_purchases: dict[int, set] = {}
    for uid, iid in val_ints:
        val_purchases.setdefault(uid, set()).add(iid)

    # Template user narrations
    user_texts = []
    for uid in range(user_features.shape[0]):
        feat = user_features[uid]
        dept_prefs = feat[-n_depts:]
        top_depts = sorted(range(n_depts), key=lambda d: -dept_prefs[d])[:3]
        desc = "Grocery shopper who buys: " + ", ".join(
            f"{dept_names[d]} ({dept_prefs[d]*100:.0f}%)" for d in top_depts
        )
        user_texts.append(desc)

    print(f"  Embedding {len(user_texts)} user profiles...")
    user_embs = model.encode(user_texts, batch_size=BATCH_SIZE, show_progress_bar=True).astype(np.float32)
    faiss.normalize_L2(user_embs)

    max_k = max(ks)
    recall: dict[int, list] = {k: [] for k in ks}
    ndcg:   dict[int, list] = {k: [] for k in ks}

    for uid, relevant in val_purchases.items():
        _, indices = index.search(user_embs[uid:uid+1], max_k)
        retrieved = indices[0]
        for k in ks:
            hits = sum(1 for iid in retrieved[:k] if int(iid) in relevant)
            recall[k].append(hits / len(relevant))
            dcg  = sum(1.0 / np.log2(r + 2) for r, iid in enumerate(retrieved[:k]) if int(iid) in relevant)
            idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
            ndcg[k].append(dcg / idcg if idcg > 0 else 0.0)

    return {
        'recall':       {k: float(np.mean(v)) for k, v in recall.items()},
        'ndcg':         {k: float(np.mean(v)) for k, v in ndcg.items()},
        'n_eval_users': len(val_purchases),
    }


def main():
    parser = argparse.ArgumentParser(description="Exp 1: LLM Item Enrichment")
    parser.add_argument('--no-llm', action='store_true', help='Use template descriptions only')
    args = parser.parse_args()

    print("\n" + "="*72)
    print("  EXPERIMENT 1: LLM ITEM ENRICHMENT")
    print("="*72)

    user_features, item_features, items, interactions = load_data()

    from sentence_transformers import SentenceTransformer
    print(f"  Loading embedding model ({EMBED_MODEL})...")
    st_model = SentenceTransformer(EMBED_MODEL)

    # Template baseline
    print("\n--- Template descriptions (no LLM) ---")
    template_descs  = {str(i): make_template_description(item) for i, item in enumerate(items)}
    template_embs   = embed_descriptions(template_descs, items, st_model)
    template_index  = build_index(template_embs)
    print("  Evaluating...")
    template_results = zero_shot_evaluate(template_index, user_features, interactions, st_model)

    # LLM-enriched
    print("\n--- LLM-enriched descriptions ---")
    llm_descs  = generate_descriptions(items, use_llm=not args.no_llm)
    llm_embs   = embed_descriptions(llm_descs, items, st_model)
    llm_index  = build_index(llm_embs)
    print("  Evaluating...")
    llm_results = zero_shot_evaluate(llm_index, user_features, interactions, st_model)

    # Summary
    print("\n" + "="*72)
    print("  EXPERIMENT 1 SUMMARY")
    print("="*72)
    print(f"\n  {'Method':<35} {'R@5':<10} {'R@10':<10} {'R@20':<10} {'N@10':<10}")
    print("  " + "-"*75)
    for label, res in [("Template zero-shot", template_results), ("LLM-enriched zero-shot", llm_results)]:
        print(f"  {label:<35} {res['recall'][5]:<10.4f} {res['recall'][10]:<10.4f} "
              f"{res['recall'][20]:<10.4f} {res['ndcg'][10]:<10.4f}")
    delta = llm_results['recall'][20] - template_results['recall'][20]
    print(f"\n  Recall@20 delta (LLM vs template): {delta:+.4f}")
    print(f"  Users evaluated: {template_results['n_eval_users']}")
    print("\n  EXPERIMENT 1 COMPLETE")


if __name__ == '__main__':
    main()
