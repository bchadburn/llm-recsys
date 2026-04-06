"""
Experiment 2: Zero-shot User Narration
Pattern: User features → text profile → sentence-transformer → item index query
         No trained UserTower needed.

Two variants:
  Template: rule-based conversion of numeric features to text
  LLM:      Claude enriches the template with inferred meal habits, household
            type, lifestyle signals

Queries the LLM-enriched FAISS index from Exp 1 (item_descriptions_cache.json).
Falls back to template item index if cache not found.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path.home() / '.env')

DATA_DIR     = Path('data/instacart')
CACHE_FILE   = Path('item_descriptions_cache.json')
MODEL        = 'claude-haiku-4-5-20251001'
EMBED_MODEL  = 'sentence-transformers/all-MiniLM-L6-v2'
N_DEMO_USERS = 2   # users per archetype group
TOP_K        = 10


def load_data():
    from data_instacart import load_instacart, DEPARTMENTS
    user_features, item_features, items, interactions, user_archetypes, up_stats = load_instacart(
        str(DATA_DIR), n_users=2000, n_items=5000, n_interactions=1500000,
    )
    return user_features, item_features, items, interactions, user_archetypes, DEPARTMENTS


def make_template_profile(user_features: np.ndarray, uid: int, dept_names: list) -> str:
    feat = user_features[uid]
    n_depts = len(dept_names)
    dept_prefs = feat[-n_depts:]
    top_depts = sorted(range(n_depts), key=lambda d: -dept_prefs[d])[:5]
    dept_str = ", ".join(
        f"{dept_names[d]} ({dept_prefs[d]*100:.0f}%)"
        for d in top_depts if dept_prefs[d] > 0.01
    )
    return f"Grocery shopper with purchase history: {dept_str}."


def make_llm_profile(client, template_profile: str) -> str:
    prompt = (
        f"Given this grocery shopper's purchase history, write a 2-sentence profile that infers "
        f"their likely meal habits, household type, and lifestyle. Be specific and vivid.\n\n"
        f"Purchase history: {template_profile}\n\n"
        f"Profile:"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=120,
        messages=[{'role': 'user', 'content': prompt}],
    )
    return response.content[0].text.strip()


def load_or_build_item_index(items: list):
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(EMBED_MODEL)

    if CACHE_FILE.exists():
        print(f"  Loading enriched item descriptions from {CACHE_FILE}...")
        cache = json.loads(CACHE_FILE.read_text())
        texts = [
            cache.get(str(item['id']),
                      f"{item['name']} - {item.get('aisle','?')} - {item.get('department','?')}")
            for item in items
        ]
        label = "LLM-enriched"
    else:
        print("  No enriched cache found — using template item descriptions.")
        texts = [
            f"{item['name']} - {item.get('aisle','?')} - {item.get('department','?')}"
            for item in items
        ]
        label = "template"

    embs = st_model.encode(texts, batch_size=256, show_progress_bar=True).astype(np.float32)
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    print(f"  Item index ready ({label}, {len(items)} items)")
    return index, st_model


def query_index(profile_text: str, st_model, index: faiss.IndexFlatIP, k: int = TOP_K) -> np.ndarray:
    emb = st_model.encode([profile_text]).astype(np.float32)
    faiss.normalize_L2(emb)
    _, indices = index.search(emb, k)
    return indices[0]


def category_precision(indices: np.ndarray, items: list, dominant_dept: str) -> float:
    hits = sum(1 for idx in indices if items[idx].get('department', '') == dominant_dept)
    return hits / len(indices)


def select_demo_users(user_archetypes: np.ndarray, n_per_group: int = N_DEMO_USERS) -> list:
    from collections import Counter
    counts = Counter(int(a) for a in user_archetypes)
    demo = []
    for arch_id, _ in counts.most_common(3):
        matches = [uid for uid, a in enumerate(user_archetypes) if int(a) == arch_id]
        demo.extend(matches[:n_per_group])
    return demo


def main():
    print("\n" + "="*72)
    print("  EXPERIMENT 2: ZERO-SHOT USER NARRATION")
    print("="*72)

    import anthropic
    client = anthropic.Anthropic()

    user_features, item_features, items, interactions, user_archetypes, dept_names = load_data()
    n_depts = len(dept_names)
    index, st_model = load_or_build_item_index(items)

    demo_users = select_demo_users(user_archetypes)
    print(f"\n  Showing results for {len(demo_users)} demo users\n")

    results_summary = []

    for uid in demo_users:
        template_profile = make_template_profile(user_features, uid, dept_names)
        llm_profile      = make_llm_profile(client, template_profile)

        template_results = query_index(template_profile, st_model, index)
        llm_results      = query_index(llm_profile, st_model, index)

        feat = user_features[uid]
        dept_prefs = feat[-n_depts:]
        dominant_dept = dept_names[int(np.argmax(dept_prefs))]

        tp = category_precision(template_results, items, dominant_dept)
        lp = category_precision(llm_results, items, dominant_dept)

        print(f"\n  User {uid}  [dominant dept: {dominant_dept}]")
        print(f"  Template profile: {template_profile}")
        print(f"  LLM profile:      {llm_profile}")
        print()
        print(f"  {'Rank':<5} {'Template results':<35} {'LLM results':<35}")
        print("  " + "-"*75)
        for rank in range(TOP_K):
            t_item = items[template_results[rank]]['name'][:32]
            l_item = items[llm_results[rank]]['name'][:32]
            print(f"  {rank+1:<5} {t_item:<35} {l_item:<35}")
        print(f"\n  Category precision@{TOP_K} (dominant={dominant_dept}): "
              f"template={tp:.2f}  LLM={lp:.2f}")

        results_summary.append({
            'uid': uid, 'dominant_dept': dominant_dept,
            'template_precision': tp, 'llm_precision': lp,
        })

    avg_t = np.mean([r['template_precision'] for r in results_summary])
    avg_l = np.mean([r['llm_precision'] for r in results_summary])

    print("\n" + "="*72)
    print("  EXPERIMENT 2 SUMMARY")
    print("="*72)
    print(f"\n  Avg category precision@{TOP_K}: template={avg_t:.3f}  LLM={avg_l:.3f}")
    print(f"  Delta (LLM vs template): {avg_l - avg_t:+.3f}")
    print(f"  Users evaluated: {len(results_summary)}")
    print("\n  EXPERIMENT 2 COMPLETE")


if __name__ == '__main__':
    main()
