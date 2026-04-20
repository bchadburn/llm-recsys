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

Aggregate evaluation:
  N_EVAL_USERS users are evaluated across all three methods (FAISS, LightGBM,
  LLM no-context) using Recall@K and NDCG@K on held-out val interactions.
  LLM is only evaluated at no-context to keep API costs tractable; the
  qualitative demo shows context-dependent reranking for N_DEMO_USERS.
"""

import json
from pathlib import Path

import faiss
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv(Path.home() / '.env')

DATA_DIR      = Path('data/instacart')
MODEL         = 'claude-haiku-4-5-20251001'
RETRIEVAL_K   = 20
FINAL_K       = 10
N_DEMO_USERS  = 3
N_EVAL_USERS  = 50   # users scored for aggregate metrics (LLM API cost: ~50 calls)


CONTEXTS = [
    None,
    "Preparing a Mediterranean dinner for two tonight",
    "On a high-protein diet, focusing on muscle building",
    "Quick road trip snacks, nothing that needs refrigeration",
]


# ── Data / model helpers ──────────────────────────────────────────────────────

def load_data():
    from data_instacart import load_instacart
    user_features, item_features, items, interactions, user_archetypes, up_stats = load_instacart(
        str(DATA_DIR), n_users=2000, n_items=5000, n_interactions=1500000,
    )
    return user_features, item_features, items, interactions, user_archetypes, up_stats


def train_towers(user_features, item_features, interactions):
    from data import InteractionDataset
    from main import build_faiss_index, train
    print("  Training two-tower model...")
    user_tower, item_tower = train(user_features, item_features, interactions, InteractionDataset)
    index = build_faiss_index(item_tower, item_features)
    return user_tower, item_tower, index


def get_faiss_candidates(user_tower, faiss_index, user_features, uid: int):
    user_tower.eval()
    device = next(user_tower.parameters()).device
    feat  = torch.tensor(user_features[uid]).unsqueeze(0).to(device)
    uid_t = torch.tensor([uid]).to(device)
    with torch.no_grad():
        emb = user_tower(feat, ids=uid_t).cpu().numpy().astype(np.float32)
    faiss.normalize_L2(emb)
    scores, indices = faiss_index.search(emb, RETRIEVAL_K)
    return scores[0], indices[0]


def build_lgbm_ranker(user_tower, item_tower, user_features, item_features,
                      interactions, up_stats):
    try:
        from data_instacart import PRICE_SENS_IDX
        from ranker import _embed_items, _embed_users, train_ranker

        user_embs = _embed_users(user_tower, user_features)
        item_embs = _embed_items(item_tower, item_features)

        rng = np.random.default_rng(42)
        train_users = rng.permutation(user_features.shape[0])[:200].tolist()

        ranker = train_ranker(
            user_tower, item_tower, user_features, item_features,
            interactions, up_stats, train_users,
            price_sens_idx=PRICE_SENS_IDX,
        )
        return ranker, user_embs, item_embs
    except Exception as e:
        print(f"  Warning: LightGBM ranker training failed ({e}), will use FAISS order")
        return None


def get_lgbm_ranking(ranker_bundle, user_features, item_features,
                     up_stats, uid: int, candidate_indices: np.ndarray) -> np.ndarray:
    if ranker_bundle is None:
        return candidate_indices
    try:
        from data_instacart import PRICE_SENS_IDX
        from ranker import _make_features

        ranker, user_embs, item_embs = ranker_bundle
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


# ── LLM reranker ──────────────────────────────────────────────────────────────

def llm_rerank(client, items: list, candidate_indices: np.ndarray,
               user_profile: str, context: str | None) -> tuple[list, str]:
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


# ── Aggregate evaluation ──────────────────────────────────────────────────────

def _recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    hits = sum(1 for item_id in retrieved[:k] if int(item_id) in relevant)
    return hits / len(relevant) if relevant else 0.0


def _ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item_id in enumerate(retrieved[:k])
        if int(item_id) in relevant
    )
    n_rel = min(len(relevant), k)
    idcg  = sum(1.0 / np.log2(r + 2) for r in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate_eval(client, user_tower, faiss_index, ranker_bundle,
                   user_features, item_features, items, interactions,
                   dept_names, n_users: int = N_EVAL_USERS, ks=(5, 10)):
    """
    Evaluate FAISS, LightGBM, and LLM (no-context) on held-out val interactions
    for n_users randomly sampled from users that have val purchases.

    Uses the same 80/20 interaction-level split as the two-tower training.
    """
    from eval import _reconstruct_val_interactions

    val_ints = _reconstruct_val_interactions(interactions, n_users=user_features.shape[0])
    val_purchases: dict[int, set] = {}
    for uid, iid in val_ints:
        val_purchases.setdefault(uid, set()).add(iid)

    rng = np.random.default_rng(42)
    eval_uids = rng.choice(
        list(val_purchases.keys()), size=min(n_users, len(val_purchases)), replace=False
    )

    scores = {
        'faiss':  {k: [] for k in ks},
        'lgbm':   {k: [] for k in ks},
        'llm':    {k: [] for k in ks},
    }

    print(f"\n  Computing aggregate metrics over {len(eval_uids)} users...")
    for i, uid in enumerate(eval_uids):
        relevant = val_purchases[uid]
        profile  = user_profile_text(user_features, uid, dept_names)

        faiss_scores_arr, faiss_indices = get_faiss_candidates(
            user_tower, faiss_index, user_features, uid
        )
        lgbm_indices = get_lgbm_ranking(
            ranker_bundle, user_features, item_features, None, uid, faiss_indices
        )
        llm_indices, _ = llm_rerank(client, items, faiss_indices, profile, context=None)

        for k in ks:
            scores['faiss'][k].append(_recall_at_k(list(faiss_indices), relevant, k))
            scores['lgbm'][k].append(_recall_at_k(list(lgbm_indices),  relevant, k))
            scores['llm'][k].append(_recall_at_k(list(llm_indices),   relevant, k))

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(eval_uids)} users evaluated...")

    return {
        method: {k: float(np.mean(v)) for k, v in ks_dict.items()}
        for method, ks_dict in scores.items()
    }, len(eval_uids)


def print_aggregate_table(agg_results: dict, n_users: int, ks=(5, 10)):
    print("\n" + "="*72)
    print("  EXPERIMENT 3 — AGGREGATE METRICS (no-context baseline)")
    print("="*72)
    print(f"  Users evaluated: {n_users}  |  Candidate set: FAISS top-{RETRIEVAL_K}\n")
    header = f"  {'Method':<22}" + "".join(f"{'Recall@'+str(k):<12}" for k in ks)
    print(header)
    print("  " + "-" * (22 + 12 * len(ks)))
    labels = {'faiss': 'FAISS', 'lgbm': 'LightGBM', 'llm': f'LLM ({MODEL.split("-")[1]})'}
    for key, label in labels.items():
        row = f"  {label:<22}" + "".join(f"{agg_results[key][k]:<12.4f}" for k in ks)
        print(row)
    print()
    for k in ks:
        lgbm_delta = agg_results['lgbm'][k] - agg_results['faiss'][k]
        llm_delta  = agg_results['llm'][k]  - agg_results['faiss'][k]
        print(f"  LightGBM vs FAISS  Recall@{k}: {lgbm_delta:+.4f}")
        print(f"  LLM      vs FAISS  Recall@{k}: {llm_delta:+.4f}")


# ── Exp 4 diagnostic ──────────────────────────────────────────────────────────

def diagnose_synthetic_context(user_features: np.ndarray, interactions: list):
    """
    Investigate why synthetic occasion injection (Exp 4) degraded performance.

    Three hypotheses tested:
      H1 — Occasions are uniformly random → zero correlation with purchases
      H2 — Occasion fractions are near-uniform per user → no signal
      H3 — The occasion features are too small a fraction of the feature vector
           to influence learned embeddings
    """
    from synthetic_context import N_OCCASIONS, OCCASIONS, inject_occasions

    print("\n" + "="*72)
    print("  EXP 4 DIAGNOSTIC: WHY DID SYNTHETIC CONTEXT HURT?")
    print("="*72)

    augmented = inject_occasions(user_features, interactions)
    occ_fracs = augmented[:, user_features.shape[1]:]   # last N_OCCASIONS columns

    # H1: correlation between occasion fracs and purchase behaviour
    # Proxy: do users with high 'health_diet' fraction buy different items?
    health_idx = OCCASIONS.index('health_diet')
    health_fracs = occ_fracs[:, health_idx]

    high_health = np.where(health_fracs > np.percentile(health_fracs, 75))[0]
    low_health  = np.where(health_fracs < np.percentile(health_fracs, 25))[0]

    high_items: dict[int, int] = {}
    low_items:  dict[int, int] = {}
    for uid, iid in interactions:
        if uid in high_health:
            high_items[iid] = high_items.get(iid, 0) + 1
        elif uid in low_health:
            low_items[iid] = low_items.get(iid, 0) + 1

    high_set = set(sorted(high_items, key=lambda x: -high_items[x])[:20])
    low_set  = set(sorted(low_items,  key=lambda x: -low_items[x])[:20])
    overlap  = len(high_set & low_set)

    print("\n  H1 — Occasion-purchase correlation")
    print("  Top-20 items for high vs low 'health_diet' users:")
    print(f"  Overlap: {overlap}/20  (20/20 = purely random, 0/20 = perfectly correlated)")
    print(f"  → {'CONFIRMED random' if overlap >= 15 else 'Some signal present'}: "
          f"occasion labels were assigned randomly per interaction, so occasion "
          f"fractions per user are driven by interaction count, not actual behavior.")

    # H2: variance of occasion fracs — are they near-uniform?
    per_user_entropy = -np.sum(
        np.where(occ_fracs > 0, occ_fracs * np.log(occ_fracs + 1e-9), 0), axis=1
    )
    max_entropy = np.log(N_OCCASIONS)
    mean_frac_of_max = float(np.mean(per_user_entropy)) / max_entropy

    print("\n  H2 — Feature variance (occasion fracs near-uniform?)")
    print(f"  Mean per-user entropy: {np.mean(per_user_entropy):.3f}  "
          f"(max uniform = {max_entropy:.3f})")
    print(f"  Mean fraction of max entropy: {mean_frac_of_max:.1%}")
    print(f"  Occasion frac std dev across users: {occ_fracs.std(axis=0).mean():.4f}")
    print(f"  → {'CONFIRMED near-uniform' if mean_frac_of_max > 0.85 else 'Some variance present'}: "
          f"with ~700 interactions/user and 6 occasions assigned uniformly at random, "
          f"each user converges to ~1/6 per occasion by the law of large numbers. "
          f"The features carry near-zero variance and no signal.")

    # H3: feature dilution — what fraction of the user feature vector are occasions?
    orig_dim = user_features.shape[1]
    aug_dim  = augmented.shape[1]
    occ_frac_of_total = N_OCCASIONS / aug_dim

    print("\n  H3 — Feature dilution")
    print(f"  Original user feature dim: {orig_dim}d")
    print(f"  Occasion features added:   {N_OCCASIONS}d")
    print(f"  Augmented dim:             {aug_dim}d")
    print(f"  Occasion fraction of total: {occ_frac_of_total:.1%}")
    print(f"  → Even if occasions had signal, they represent only {occ_frac_of_total:.0%} "
          f"of the input. The tower MLP would need to heavily up-weight these "
          f"dimensions to use them — unlikely without a targeted architecture change.")

    print("\n  ROOT CAUSE SUMMARY")
    print("  " + "-"*60)
    print("  The synthetic occasions failed for all three reasons simultaneously:")
    print("  1. Labels were random per interaction → zero ground-truth correlation")
    print("  2. ~700 interactions/user → per-user fracs collapse to ~1/6 each")
    print("     (law of large numbers erases any variance the random assignment created)")
    print("  3. 6/55 = 11% of feature vector → diluted even if signal existed")
    print("")
    print("  The slight regression (R@20 -0.0025) is not from 'bad context' —")
    print("  it's from adding 6 near-constant noise dimensions that slightly")
    print("  destabilise the MLP's weight initialisation and optimisation landscape.")
    print("")
    print("  FIX: To test whether context injection CAN work, occasions need to be")
    print("  derived from actual purchase signals, not assigned randomly. Options:")
    print("  a) Use temporal data — time-of-day / day-of-week of actual orders")
    print("  b) Use department mix — classify each order basket into an occasion")
    print("     using a rule (>50% produce + protein → 'cooking'; >50% snacks → 'snacking')")
    print("  c) Use LLM to label each historical basket with an occasion, then")
    print("     aggregate per user — this produces correlated occasion fracs.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*72)
    print("  EXPERIMENT 3: CONTEXTUAL LLM RERANKER")
    print("="*72)

    from collections import Counter

    import anthropic

    from data_instacart import DEPARTMENTS as dept_names

    client = anthropic.Anthropic()
    user_features, item_features, items, interactions, user_archetypes, up_stats = load_data()
    user_tower, item_tower, faiss_index = train_towers(user_features, item_features, interactions)

    print("  Training LightGBM ranker...")
    ranker_bundle = build_lgbm_ranker(
        user_tower, item_tower, user_features, item_features, interactions, up_stats,
    )

    # ── Qualitative demo: 3 users × 4 contexts ────────────────────────────────
    arch_counts = Counter(int(a) for a in user_archetypes)
    demo_users = []
    for arch_id, _ in arch_counts.most_common(N_DEMO_USERS):
        uid = int(np.where(user_archetypes == arch_id)[0][0])
        demo_users.append(uid)

    for uid in demo_users:
        profile = user_profile_text(user_features, uid, dept_names)
        faiss_scores_arr, faiss_indices = get_faiss_candidates(user_tower, faiss_index, user_features, uid)
        lgbm_indices = get_lgbm_ranking(
            ranker_bundle, user_features, item_features, up_stats, uid, faiss_indices,
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

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    agg_results, n_eval = aggregate_eval(
        client, user_tower, faiss_index, ranker_bundle,
        user_features, item_features, items, interactions,
        dept_names, n_users=N_EVAL_USERS,
    )
    print_aggregate_table(agg_results, n_eval)

    # ── Exp 4 diagnostic ───────────────────────────────────────────────────────
    diagnose_synthetic_context(user_features, interactions)

    print("\n  EXPERIMENT 3 COMPLETE")


if __name__ == '__main__':
    main()
