"""
LLM item description quality evaluations.

Metrics (all model-free — no additional API calls required):
  - department_coverage   : fraction of descriptions that mention the item's department/aisle
  - persona_diversity     : number of unique persona terms across all descriptions
  - avg_token_count       : mean word count (proxy for description richness)
  - template_vs_llm_length: word-count ratio LLM/template (>1 means richer)
"""

import json
import re
from pathlib import Path
from typing import Any

# Persona/shopper terms to scan for (lower-cased)
PERSONA_TERMS = [
    "parent", "family", "athlete", "fitness", "vegan", "vegetarian", "keto",
    "gluten", "budget", "gourmet", "busy", "health", "student", "elderly",
    "child", "toddler", "diabetic", "organic", "convenience", "meal prep",
    "snack", "breakfast", "dinner", "lunch", "entertaining", "party",
    "muscle", "weight", "diet", "cook", "baker", "brunch",
]


def _word_count(text: str) -> int:
    return len(text.split())


def _mentions_department(description: str, item: dict) -> bool:
    """True if the description contains any word from the item's department or aisle."""
    desc_lower = description.lower()
    for field in ("department", "aisle"):
        value = item.get(field, "")
        if not value or value == "unknown":
            continue
        # Split multi-word department names and check each word (≥4 chars to avoid stop words)
        for word in re.split(r"[\s/,]+", value.lower()):
            if len(word) >= 4 and word in desc_lower:
                return True
    return False


def evaluate_descriptions(
    descriptions: dict[str, str],
    items: list[dict],
    template_descriptions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Args:
        descriptions: {str(item_idx): description_text} — LLM or template descriptions
        items: list of item dicts (same order as enumerate index)
        template_descriptions: optional template descriptions for length comparison

    Returns:
        dict of scalar metrics
    """
    dept_hits = 0
    persona_hit_counts: dict[str, int] = {t: 0 for t in PERSONA_TERMS}
    word_counts: list[int] = []
    template_word_counts: list[int] = []

    for idx_str, desc in descriptions.items():
        idx = int(idx_str)
        item = items[idx] if idx < len(items) else {}

        if _mentions_department(desc, item):
            dept_hits += 1

        desc_lower = desc.lower()
        for term in PERSONA_TERMS:
            if term in desc_lower:
                persona_hit_counts[term] += 1

        word_counts.append(_word_count(desc))

        if template_descriptions and idx_str in template_descriptions:
            template_word_counts.append(_word_count(template_descriptions[idx_str]))

    n = len(descriptions)
    unique_personas_used = sum(1 for c in persona_hit_counts.values() if c > 0)

    results: dict[str, Any] = {
        "n_descriptions": n,
        "department_coverage": round(dept_hits / n, 4) if n else 0.0,
        "persona_diversity": unique_personas_used,
        "persona_coverage_pct": round(unique_personas_used / len(PERSONA_TERMS), 4),
        "avg_word_count": round(sum(word_counts) / n, 1) if n else 0.0,
        "top_persona_terms": sorted(
            persona_hit_counts.items(), key=lambda x: -x[1]
        )[:10],
    }

    if template_word_counts:
        avg_llm = sum(word_counts) / n
        avg_tmpl = sum(template_word_counts) / len(template_word_counts)
        results["template_vs_llm_length_ratio"] = round(avg_llm / avg_tmpl, 3) if avg_tmpl else None

    return results


def load_and_evaluate(
    cache_file: Path,
    items: list[dict],
    template_fn=None,
) -> dict[str, Any]:
    """Convenience wrapper that loads the description cache and runs eval."""
    if not cache_file.exists():
        raise FileNotFoundError(f"Description cache not found: {cache_file}")
    descriptions = json.loads(cache_file.read_text())

    template_descriptions = None
    if template_fn is not None:
        template_descriptions = {str(i): template_fn(item) for i, item in enumerate(items)}

    return evaluate_descriptions(descriptions, items, template_descriptions)
