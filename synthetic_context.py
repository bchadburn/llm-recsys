"""
Experiment 4: Synthetic Context Injection
Pattern: Augment training interactions with synthetic occasions → context-aware two-tower

Appends a one-hot occasion vector (6-dim) to user features (49d → 55d).
Occasion is assigned pseudo-randomly per interaction (SEED=42).
Hypothesis: neutral/slight improvement since occasions don't correlate with
real purchase behavior, but establishes the training pipeline for future
experiments where context IS meaningful.
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path('data/instacart')

OCCASIONS = [
    'weeknight_dinner',
    'meal_prep',
    'party_hosting',
    'health_diet',
    'quick_snack',
    'road_trip',
]
N_OCCASIONS = len(OCCASIONS)
SEED = 42


def load_data():
    from data_instacart import load_instacart
    print("Loading Instacart data...")
    return load_instacart(str(DATA_DIR), n_users=2000, n_items=5000, n_interactions=1500000)


def inject_occasions(user_features: np.ndarray, interactions: list, seed: int = SEED):
    """
    Assign a random occasion to each interaction, aggregate per user as
    occasion-frequency fractions, and append to user feature vectors.

    Returns:
        augmented_user_features: (n_users, original_dim + N_OCCASIONS)
    """
    rng = np.random.default_rng(seed)
    n_users = user_features.shape[0]

    occasion_indices = rng.integers(0, N_OCCASIONS, size=len(interactions))

    occasion_counts = np.zeros((n_users, N_OCCASIONS), dtype=np.float32)
    for (uid, _), occ_idx in zip(interactions, occasion_indices):
        occasion_counts[uid, occ_idx] += 1

    row_sums = occasion_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    occasion_fracs = occasion_counts / row_sums

    augmented = np.concatenate([user_features, occasion_fracs], axis=1).astype(np.float32)
    return augmented


def run_training(user_features, item_features, interactions, label: str):
    from data import InteractionDataset
    from main import build_faiss_index, train
    print(f"\n  Training two-tower ({label}, user_dim={user_features.shape[1]})...")
    user_tower, item_tower = train(user_features, item_features, interactions, InteractionDataset)
    index = build_faiss_index(item_tower, item_features)
    return user_tower, item_tower, index


def run_eval(user_tower, index, user_features, interactions, ks=(5, 10, 20)):
    from eval import evaluate
    return evaluate(user_tower, index, user_features, interactions, ks=ks)


def main():
    print("\n" + "="*72)
    print("  EXPERIMENT 4: SYNTHETIC CONTEXT INJECTION")
    print("="*72)

    user_features, item_features, items, interactions, user_archetypes, up_stats = load_data()

    # Baseline: no context
    print("\n--- Baseline: no synthetic context ---")
    user_tower_base, _, index_base = run_training(user_features, item_features, interactions, "baseline")
    base_results = run_eval(user_tower_base, index_base, user_features, interactions)

    # Context-augmented
    print("\n--- Context-augmented: synthetic occasions injected ---")
    augmented_features = inject_occasions(user_features, interactions)
    print(f"  User feature dim: {user_features.shape[1]}d → {augmented_features.shape[1]}d")
    print(f"  Occasions: {', '.join(OCCASIONS)}")
    user_tower_ctx, _, index_ctx = run_training(augmented_features, item_features, interactions, "context-augmented")
    ctx_results = run_eval(user_tower_ctx, index_ctx, augmented_features, interactions)

    # Summary
    print("\n" + "="*72)
    print("  EXPERIMENT 4 SUMMARY")
    print("="*72)
    print(f"\n  {'Method':<30} {'R@5':<10} {'R@10':<10} {'R@20':<10} {'N@10':<10}")
    print("  " + "-"*60)
    for label, res in [("Baseline (no context)", base_results), ("Synthetic occasions", ctx_results)]:
        print(f"  {label:<30} {res['recall'][5]:<10.4f} {res['recall'][10]:<10.4f} "
              f"{res['recall'][20]:<10.4f} {res['ndcg'][10]:<10.4f}")

    delta_r20 = ctx_results['recall'][20] - base_results['recall'][20]
    print(f"\n  Recall@20 delta (context vs baseline): {delta_r20:+.4f}")
    print(f"  Users evaluated: {base_results['n_eval_users']}")
    print("\n  EXPERIMENT 4 COMPLETE")


if __name__ == '__main__':
    main()
