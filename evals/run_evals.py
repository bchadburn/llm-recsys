"""
LLM eval pipeline for the recommendation system.

Runs model-free quality checks on cached LLM outputs (no new API calls
unless --reranker-live is passed). Results are written to evals/results/
as timestamped JSON files so regressions are visible across runs.

Usage:
    # descriptions only (fast, reads item_descriptions_cache.json)
    uv run --with python-dotenv python evals/run_evals.py --descriptions

    # reranker quality check on N sample users (uses LLM API)
    uv run --with anthropic --with python-dotenv --with faiss-cpu \\
        --with lightgbm --with xgboost --with torch \\
        python evals/run_evals.py --reranker --reranker-users 20

    # full run
    uv run --with anthropic --with python-dotenv --with faiss-cpu \\
        --with lightgbm --with xgboost --with torch \\
        python evals/run_evals.py --descriptions --reranker

    # compare results across runs
    python evals/run_evals.py --compare
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add repo root to path so sibling modules are importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.prompt_registry import register_all
from evals.description_evals import load_and_evaluate
from evals.reranker_evals import (
    score_single_response,
    aggregate_scores,
    context_sensitivity,
)

RESULTS_DIR   = Path(__file__).parent / "results"
CACHE_FILE    = Path("item_descriptions_cache.json")
DATA_DIR      = Path("data/instacart")
RETRIEVAL_K   = 20
FINAL_K       = 10
CONTEXTS      = [
    None,
    "Preparing a Mediterranean dinner for two tonight",
    "On a high-protein diet, focusing on muscle building",
    "Quick road trip snacks, nothing that needs refrigeration",
]


# ── Result persistence ────────────────────────────────────────────────────────

def save_results(results: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"eval_{ts}.json"
    out.write_text(json.dumps(results, indent=2))
    return out


def load_all_results() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    return [
        json.loads(f.read_text())
        for f in sorted(RESULTS_DIR.glob("eval_*.json"))
    ]


# ── Description eval ──────────────────────────────────────────────────────────

def run_description_eval(items: list[dict]) -> dict:
    print("\n── Description quality eval ─────────────────────────────────────")
    if not CACHE_FILE.exists():
        print(f"  [SKIP] Cache not found: {CACHE_FILE}")
        print("  Run llm_item_enrichment.py first to generate descriptions.")
        return {}

    from llm_item_enrichment import make_template_description
    metrics = load_and_evaluate(CACHE_FILE, items, template_fn=make_template_description)

    print(f"  Descriptions evaluated : {metrics['n_descriptions']}")
    print(f"  Department coverage    : {metrics['department_coverage']:.1%}")
    print(f"  Persona diversity      : {metrics['persona_diversity']} / {len(metrics['top_persona_terms'])} tracked terms")
    print(f"  Avg word count         : {metrics['avg_word_count']:.1f} words")
    if "template_vs_llm_length_ratio" in metrics:
        print(f"  LLM/template length    : {metrics['template_vs_llm_length_ratio']:.2f}x")
    print(f"  Top persona terms      : {', '.join(t for t,_ in metrics['top_persona_terms'][:5])}")

    return {"description_eval": metrics}


# ── Reranker eval ─────────────────────────────────────────────────────────────

def run_reranker_eval(n_users: int, items: list[dict], user_features, item_features,
                      interactions: list) -> dict:
    print("\n── Reranker quality eval ────────────────────────────────────────")

    import anthropic
    import faiss
    import numpy as np
    import torch
    from dotenv import load_dotenv
    from pathlib import Path as _Path

    load_dotenv(_Path.home() / ".env")

    from main import train, build_faiss_index
    from lgbm_ranker import train_lgbm_ranker
    from llm_reranker import (
        get_faiss_candidates, get_lgbm_ranking, llm_rerank, user_profile_text,
    )
    from eval import _reconstruct_val_interactions
    from data_instacart import load_instacart

    client = anthropic.Anthropic()

    # Load dept names (last N features of user_features are dept fractions)
    raw = load_instacart(str(DATA_DIR), n_users=2000, n_items=5000, n_interactions=1500000)
    dept_names = [d for d in raw[2][0].keys() if d not in
                  ("name", "aisle", "category", "cat_idx", "price_tier", "popularity", "features")]
    # Fallback: derive dept names from item dicts
    all_depts = sorted({item.get("department", "unknown") for item in items})

    # Train two-tower
    print(f"  Training two-tower model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    user_tower, item_tower = train(user_features, item_features, interactions, device=device)
    faiss_index = build_faiss_index(item_tower, item_features, device=device)

    # Train LightGBM ranker
    ranker_bundle = train_lgbm_ranker(user_features, item_features, interactions)

    # Sample eval users with val purchases
    val_ints = _reconstruct_val_interactions(interactions, n_users=user_features.shape[0])
    val_purchases: dict[int, set] = {}
    for uid, iid in val_ints:
        val_purchases.setdefault(uid, set()).add(iid)

    rng = np.random.default_rng(42)
    eval_uids = rng.choice(
        list(val_purchases.keys()), size=min(n_users, len(val_purchases)), replace=False
    )

    all_scores: list[dict] = []
    ctx_sensitivity: list[dict] = []

    print(f"  Evaluating {len(eval_uids)} users × {len(CONTEXTS)} contexts...")
    for i, uid in enumerate(eval_uids):
        profile = user_profile_text(user_features, uid, all_depts)
        _, faiss_indices = get_faiss_candidates(user_tower, faiss_index, user_features, uid)

        no_ctx_ranking, _ = llm_rerank(client, items, faiss_indices, profile, context=None)

        ctx_rankings = {}
        for ctx in CONTEXTS[1:]:
            ctx_ranking, _ = llm_rerank(client, items, faiss_indices, profile, context=ctx)
            ctx_rankings[ctx[:30]] = ctx_ranking

            reranked, reasoning = llm_rerank(client, items, faiss_indices, profile, context=ctx)
            score = score_single_response(
                ranking=reranked,
                reasoning=reasoning,
                candidate_indices=list(faiss_indices),
                items=items,
                faiss_ranking=list(faiss_indices),
            )
            all_scores.append(score)

        # No-context response scored separately
        nc_reranked, nc_reasoning = llm_rerank(client, items, faiss_indices, profile, context=None)
        nc_score = score_single_response(
            ranking=nc_reranked,
            reasoning=nc_reasoning,
            candidate_indices=list(faiss_indices),
            items=items,
            faiss_ranking=list(faiss_indices),
        )
        all_scores.append(nc_score)

        ctx_sensitivity.append(context_sensitivity(no_ctx_ranking, ctx_rankings))

        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(eval_uids)} users done")

    agg = aggregate_scores(all_scores)
    mean_ctx_sensitivity = (
        sum(d.get("mean", 0) for d in ctx_sensitivity) / len(ctx_sensitivity)
        if ctx_sensitivity else 0.0
    )

    print(f"\n  Responses evaluated         : {agg['n_responses']}")
    print(f"  JSON parse failure rate     : {agg['json_parse_failure_rate']:.1%}")
    print(f"  Reasoning alignment         : {agg['reasoning_alignment']:.1%}")
    print(f"  Hallucination rate          : {agg['hallucination_rate']:.1%}")
    print(f"  Avg Kendall τ vs FAISS      : {agg['avg_kendall_tau_dist']:.3f}")
    print(f"  Avg FAISS overlap@5         : {agg['avg_faiss_overlap_at_k']:.3f}")
    print(f"  Mean context sensitivity τ  : {mean_ctx_sensitivity:.3f}")
    print(f"  Avg reasoning words         : {agg['avg_reasoning_words']:.1f}")

    return {
        "reranker_eval": {
            **agg,
            "mean_context_sensitivity": round(mean_ctx_sensitivity, 4),
            "n_users": len(eval_uids),
        }
    }


# ── Comparison across runs ────────────────────────────────────────────────────

def print_comparison() -> None:
    runs = load_all_results()
    if not runs:
        print("No saved eval results found in evals/results/")
        return

    print(f"\n{'Run':<26} {'Dept Cov':>9} {'Persona Div':>12} {'Align':>7} {'Halluc':>7} {'Ctx-τ':>6}")
    print("-" * 72)
    for r in runs:
        ts  = r.get("timestamp", "?")[:19]
        d   = r.get("description_eval", {})
        re_ = r.get("reranker_eval", {})
        print(
            f"{ts:<26}"
            f"{d.get('department_coverage', float('nan')):>9.1%}"
            f"{d.get('persona_diversity', '-'):>12}"
            f"{re_.get('reasoning_alignment', float('nan')):>7.1%}"
            f"{re_.get('hallucination_rate', float('nan')):>7.1%}"
            f"{re_.get('mean_context_sensitivity', float('nan')):>6.3f}"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM quality evals for the recommendation system")
    parser.add_argument("--descriptions", action="store_true", help="Run description quality eval")
    parser.add_argument("--reranker",     action="store_true", help="Run reranker quality eval (uses LLM API)")
    parser.add_argument("--reranker-users", type=int, default=10, metavar="N",
                        help="Number of users to evaluate for reranker (default: 10)")
    parser.add_argument("--compare",      action="store_true", help="Print comparison table of past runs")
    args = parser.parse_args()

    if args.compare:
        print_comparison()
        return

    if not args.descriptions and not args.reranker:
        parser.print_help()
        return

    # Register canonical prompts
    prompt_versions = register_all()
    print(f"Prompt versions: {prompt_versions}")

    # Load data once (shared across evals)
    from data_instacart import load_instacart
    print("\nLoading Instacart data...")
    user_features, item_features, items, interactions, _, _ = load_instacart(
        str(DATA_DIR), n_users=2000, n_items=5000, n_interactions=1500000,
    )
    print(f"  {len(items)} items | {len(interactions)} interactions")

    results: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_versions": prompt_versions,
    }

    if args.descriptions:
        results.update(run_description_eval(items))

    if args.reranker:
        results.update(run_reranker_eval(
            args.reranker_users, items, user_features, item_features, interactions
        ))

    out = save_results(results)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
