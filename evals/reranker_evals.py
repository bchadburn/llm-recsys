"""
LLM reranker quality evaluations.

Metrics (all model-free):
  - reasoning_alignment      : fraction of responses where ≥1 top-3 item name is
                               mentioned in the reasoning text
  - hallucination_rate       : fraction of responses where reasoning mentions a name
                               not in the candidate list
  - json_parse_failure_rate  : fraction of LLM responses that failed JSON parsing
  - context_sensitivity      : mean Kendall's τ distance between no-context ranking
                               and each context ranking (higher = more context-aware)
  - faiss_overlap_at_k       : mean overlap@K between LLM ranking and FAISS baseline
                               (lower = LLM is making meaningfully different choices)
"""

from __future__ import annotations

import re
from typing import Any

# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    """Lowercase, strip punctuation for fuzzy name matching."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def _reasoning_mentions(reasoning: str, item_names: list[str]) -> list[str]:
    """Return which item names appear (fuzzy) in reasoning text."""
    r = _normalize(reasoning)
    return [n for n in item_names if _normalize(n) in r]


def _kendall_tau_distance(rank_a: list[int], rank_b: list[int]) -> float:
    """
    Normalised Kendall tau distance between two rankings of the same items.
    Returns 0 if identical, 1 if fully reversed.
    Only considers items that appear in both rankings.
    """
    common = [x for x in rank_a if x in rank_b]
    if len(common) < 2:
        return 0.0
    pos_a = {item: i for i, item in enumerate(common)}
    pos_b = {item: rank_b.index(item) for item in common}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a_sign = pos_a[common[i]] - pos_a[common[j]]
            b_sign = pos_b[common[i]] - pos_b[common[j]]
            if a_sign * b_sign > 0:
                concordant += 1
            elif a_sign * b_sign < 0:
                discordant += 1
    total = concordant + discordant
    return discordant / total if total > 0 else 0.0


def _overlap_at_k(list_a: list[int], list_b: list[int], k: int) -> float:
    set_a = set(list_a[:k])
    set_b = set(list_b[:k])
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / k


# ── Per-response evaluation ───────────────────────────────────────────────────

def score_single_response(
    ranking: list[int],           # item indices in LLM order (0-based, FAISS item ids)
    reasoning: str,
    candidate_indices: list[int], # FAISS top-K candidates passed to the LLM
    items: list[dict],
    faiss_ranking: list[int],     # original FAISS order for the same candidates
    k: int = 5,
) -> dict[str, Any]:
    """Score one reranker response."""
    parse_failed = reasoning == "(JSON parsing failed)"

    candidate_names = [items[idx]["name"] for idx in candidate_indices]
    top_k_names     = [items[idx]["name"] for idx in ranking[:3] if idx < len(items)]

    non_candidate   = [
        n for n in _reasoning_mentions(reasoning, [i["name"] for i in items])
        if n not in [_normalize(c) for c in candidate_names]
    ]
    aligned         = any(
        _normalize(n) in _normalize(reasoning) for n in top_k_names
    )

    tau_dist        = _kendall_tau_distance(ranking, faiss_ranking)
    overlap         = _overlap_at_k(ranking, faiss_ranking, k)

    return {
        "parse_failed":        parse_failed,
        "reasoning_aligned":   aligned and not parse_failed,
        "hallucinated_names":  non_candidate,
        "has_hallucination":   len(non_candidate) > 0,
        "kendall_tau_dist":    tau_dist,
        "faiss_overlap_at_k":  overlap,
        "reasoning_word_count": len(reasoning.split()) if not parse_failed else 0,
    }


# ── Aggregate over multiple responses ────────────────────────────────────────

def aggregate_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of per-response score dicts into summary statistics."""
    n = len(scores)
    if n == 0:
        return {}

    def mean(key: str) -> float:
        vals = [s[key] for s in scores if isinstance(s[key], (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "n_responses":             n,
        "json_parse_failure_rate": round(sum(s["parse_failed"] for s in scores) / n, 4),
        "reasoning_alignment":     round(sum(s["reasoning_aligned"] for s in scores) / n, 4),
        "hallucination_rate":      round(sum(s["has_hallucination"] for s in scores) / n, 4),
        "avg_kendall_tau_dist":    mean("kendall_tau_dist"),
        "avg_faiss_overlap_at_k":  mean("faiss_overlap_at_k"),
        "avg_reasoning_words":     mean("reasoning_word_count"),
    }


# ── Context-sensitivity analysis ─────────────────────────────────────────────

def context_sensitivity(
    no_context_ranking: list[int],
    context_rankings: dict[str, list[int]],
) -> dict[str, float]:
    """
    Measure how much each context shifts the ranking vs. the no-context baseline.

    Args:
        no_context_ranking: item indices in LLM order with no context
        context_rankings: {context_label: ranked_item_indices}

    Returns:
        {context_label: kendall_tau_distance, "mean": mean_distance}
    """
    distances = {}
    for label, ranking in context_rankings.items():
        distances[label] = round(_kendall_tau_distance(no_context_ranking, ranking), 4)
    if distances:
        distances["mean"] = round(sum(distances.values()) / len(distances), 4)
    return distances
