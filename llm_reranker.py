"""
Experiment 3: Contextual LLM Reranker
Pattern: FAISS top-20 + session context → Claude reranking with reasoning

Demonstrates what LLMs uniquely enable: context-aware reranking that dot-product
similarity fundamentally cannot express. Same user, different context → different
ranking.

Contexts tested:
  - No context (baseline)
  - Mediterranean dinner for two
  - High-protein / muscle-building diet
  - Road trip snacks (non-refrigerated)
"""

import json
import numpy as np
import torch
import faiss
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path.home() / '.env')

DATA_DIR     = Path('data/instacart')
MODEL        = 'claude-haiku-4-5-20251001'
RETRIEVAL_K  = 20
FINAL_K      = 10
N_DEMO_USERS = 3

CONTEXTS = [
    None,
    "Preparing a Mediterranean dinner for two tonight",
    "On a high-protein diet, focusing on muscle building",
    "Quick road trip snacks, nothing that needs refrigeration",
]


def load_data():
    from data_instacart import load_instacart
    user_features, item_features, items, interactions, user_archetypes, up_stats = load_instacart(
        str(DATA_DIR), n_users=2000, n_items=5000, n_interactions=1500000,
    )
    return user_features, item_features, items, interactions, user_archetypes, up_stats


def train_towers(user_features, item_features, interactions):
    from main import train, build_faiss_index
    from data import InteractionDataset
    print("  Training two-tower model...")
    user_tower, item_tower = train(user_features, item_features, interactions, InteractionDataset)
    index = build_faiss_index(item_tower, item_features)
    return user_tower, item_tower, index


def get_faiss_candidates(user_tower, faiss_index, user_features, uid: int):
    user_tower.eval()
    feat  = torch.tensor(user_features[uid]).unsqueeze(0)
    uid_t = torch.tensor([uid])
    with torch.no_grad():
        emb = user_tower(feat, ids=uid_t).numpy().astype(np.float32)
    faiss.normalize_L2(emb)
    scores, indices = faiss_index.search(emb, RETRIEVAL_K)
    return scores[0], indices[0]


def get_lgbm_ranking(user_tower, item_tower, faiss_index,
                     user_features, item_features, interactions, up_stats,
                     uid: int, candidate_indices: np.ndarray) -> np.ndarray:
    """Re-rank candidates with LightGBM. Falls back to FAISS order on any error."""
    try:
        from ranker import train_ranker, _make_features, _embed_users, _embed_items
        from data_instacart import PRICE_SENS_IDX

        user_embs = _embed_users(user_tower, user_features)
        item_embs = _embed_items(item_tower, item_features)

        rng = np.random.default_rng(42)
        train_users = rng.permutation(user_features.shape[0])[:200].tolist()

        ranker = train_ranker(
            user_tower, item_tower, user_features, item_features,
            interactions, up_stats, train_users,
            price_sens_idx=PRICE_SENS_IDX,
        )

        u_emb = user_embs[uid]
        faiss_scores = item_embs[candidate_indices] @ u_emb
        features = _make_features(
            u_emb, item_embs[candidate_indices], faiss_scores, candidate_indices,
            user_features[uid], item_features[candidate_indices],
            up_stats, uid, PRICE_SENS_IDX,
        )
        scores = ranker.predict(features)
        return candidate_indices[np.argsort(-scores)]
    except Exception as e:
        print(f"    Warning: LightGBM reranking failed ({e}), using FAISS order")
        return candidate_indices


def llm_rerank(client, items: list, candidate_indices: np.ndarray,
               user_profile: str, context: str | None) -> tuple[list, str]:
    """Ask Claude to rerank candidates given optional shopping context."""
    candidate_list = "\n".join(
        f"{i+1}. {items[idx]['name']} ({items[idx].get('department','?')})"
        for i, idx in enumerate(candidate_indices[:RETRIEVAL_K])
    )
    context_line = (f"Shopping context: {context}"
                    if context else "No specific shopping context (general recommendations).")

    prompt = (
        f"You are a grocery recommendation system. Rerank the following {len(candidate_indices)} "
        f"candidate items for a shopper.\n\n"
        f"Shopper profile: {user_profile}\n"
        f"{context_line}\n\n"
        f"Candidates:\n{candidate_list}\n\n"
        f"Return ONLY a JSON object with two keys:\n"
        f"  'ranking': list of candidate numbers (1-{len(candidate_indices)}) "
        f"in your preferred order, top {FINAL_K} only\n"
        f"  'reasoning': 1-2 sentences explaining your top 3 choices\n\n"
        f"JSON:"
    )
    response = client.messages.create(
        model=MODEL, max_tokens=300,
        messages=[{'role': 'user', 'content': prompt}],
    )
    text = response.content[0].text.strip()
    try:
        start  = text.find('{')
        end    = text.rfind('}') + 1
        parsed = json.loads(text[start:end])
        ranking   = [int(r) - 1 for r in parsed['ranking'][:FINAL_K]]
        reranked  = [candidate_indices[r] for r in ranking if r < len(candidate_indices)]
        reasoning = parsed.get('reasoning', '')
        return reranked, reasoning
    except Exception:
        return list(candidate_indices[:FINAL_K]), "(JSON parsing failed)"


def user_profile_text(user_features: np.ndarray, uid: int, dept_names: list) -> str:
    feat = user_features[uid]
    n_depts = len(dept_names)
    dept_prefs = feat[-n_depts:]
    top = sorted(range(n_depts), key=lambda d: -dept_prefs[d])[:3]
    return "Buys: " + ", ".join(
        f"{dept_names[d]} ({dept_prefs[d]*100:.0f}%)" for d in top
    )


def main():
    print("\n" + "="*72)
    print("  EXPERIMENT 3: CONTEXTUAL LLM RERANKER")
    print("="*72)

    import anthropic
    from data_instacart import DEPARTMENTS as dept_names
    from collections import Counter

    client = anthropic.Anthropic()
    user_features, item_features, items, interactions, user_archetypes, up_stats = load_data()
    user_tower, item_tower, faiss_index = train_towers(user_features, item_features, interactions)

    # Pick 3 demo users from the 3 largest archetype groups
    arch_counts = Counter(int(a) for a in user_archetypes)
    demo_users = []
    for arch_id, _ in arch_counts.most_common(N_DEMO_USERS):
        uid = int(np.where(user_archetypes == arch_id)[0][0])
        demo_users.append(uid)

    for uid in demo_users:
        profile = user_profile_text(user_features, uid, dept_names)
        faiss_scores, faiss_indices = get_faiss_candidates(user_tower, faiss_index, user_features, uid)
        lgbm_indices = get_lgbm_ranking(
            user_tower, item_tower, faiss_index,
            user_features, item_features, interactions, up_stats,
            uid, faiss_indices,
        )

        print(f"\n{'='*100}")
        print(f"  User {uid}  —  {profile}")
        print(f"{'='*100}")

        for context in CONTEXTS:
            ctx_label = context if context else "No context (baseline)"
            llm_indices, reasoning = llm_rerank(client, items, faiss_indices, profile, context)

            print(f"\n  Context: {ctx_label}")
            if reasoning:
                print(f"  Reasoning: {reasoning}")
            print()
            print(f"  {'Rank':<5} {'FAISS':<33} {'LightGBM':<33} {'LLM-reranked':<33}")
            print("  " + "-"*104)
            for rank in range(FINAL_K):
                f_item  = items[faiss_indices[rank]]['name'][:30] if rank < len(faiss_indices) else ""
                l_item  = items[lgbm_indices[rank]]['name'][:30]  if rank < len(lgbm_indices)  else ""
                ll_item = items[llm_indices[rank]]['name'][:30]   if rank < len(llm_indices)   else ""
                print(f"  {rank+1:<5} {f_item:<33} {l_item:<33} {ll_item:<33}")

    print("\n  EXPERIMENT 3 COMPLETE")


if __name__ == '__main__':
    main()
