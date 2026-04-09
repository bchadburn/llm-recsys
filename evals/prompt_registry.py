"""
Prompt registry — version-tracks prompts by content hash so eval results
can be linked to the exact prompt text that produced them.

Usage:
    registry = PromptRegistry.load()
    version = registry.register("item_description", ITEM_DESCRIPTION_PROMPT)
    registry.save()
    # version == "v1" first time, "v2" if prompt changed, etc.
"""

import hashlib
import json
from pathlib import Path

REGISTRY_FILE = Path(__file__).parent / "prompt_versions.json"

# Canonical prompt templates — single source of truth for evals.
# Update these if the prompts in the experiment scripts change.
ITEM_DESCRIPTION_PROMPT = (
    "Write a 2-sentence product description for a grocery item that would help a "
    "recommendation system understand what kind of shopper buys it and what meal occasions "
    "it fits. Be specific about use cases and co-purchase context.\n\n"
    "Product: {name}\n"
    "Aisle: {aisle}\n"
    "Department: {department}\n\n"
    "Description:"
)

USER_NARRATION_PROMPT = (
    "Given this grocery shopper's purchase history, write a 2-sentence profile that infers "
    "their likely meal habits, household type, and lifestyle. Be specific and vivid.\n\n"
    "Purchase history: {template_profile}"
)

RERANKER_PROMPT = (
    "You are a grocery recommendation system. Rerank the following {n_candidates} "
    "candidate items for a shopper.\n\n"
    "Shopper profile: {user_profile}\n"
    "{context_line}\n\n"
    "Candidates:\n{candidate_list}\n\n"
    "Return ONLY a JSON object with two keys:\n"
    "  'ranking': list of candidate numbers (1-{n_candidates}) in your preferred order, top {final_k} only\n"
    "  'reasoning': 1-2 sentences explaining your top 3 choices\n\n"
    "JSON:"
)

PROMPTS: dict[str, str] = {
    "item_description": ITEM_DESCRIPTION_PROMPT,
    "user_narration": USER_NARRATION_PROMPT,
    "reranker": RERANKER_PROMPT,
}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:10]


class PromptRegistry:
    def __init__(self, data: dict):
        # data: {prompt_name: [{hash, version, snippet}]}
        self._data = data

    @classmethod
    def load(cls) -> "PromptRegistry":
        if REGISTRY_FILE.exists():
            return cls(json.loads(REGISTRY_FILE.read_text()))
        return cls({})

    def save(self) -> None:
        REGISTRY_FILE.write_text(json.dumps(self._data, indent=2))

    def register(self, name: str, text: str) -> str:
        """Register a prompt and return its version string (e.g. 'v1', 'v2')."""
        h = _hash(text)
        history = self._data.setdefault(name, [])
        for entry in history:
            if entry["hash"] == h:
                return entry["version"]
        version = f"v{len(history) + 1}"
        history.append({
            "hash": h,
            "version": version,
            "snippet": text[:120].replace("\n", " "),
        })
        return version

    def current_versions(self) -> dict[str, str]:
        """Return {name: version} for all registered prompts."""
        return {
            name: entries[-1]["version"]
            for name, entries in self._data.items()
            if entries
        }


def register_all() -> dict[str, str]:
    """Register all canonical prompts and return current version map."""
    registry = PromptRegistry.load()
    for name, text in PROMPTS.items():
        registry.register(name, text)
    registry.save()
    return registry.current_versions()
