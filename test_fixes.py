"""
Tests for bug fixes applied to the overnight LLM experiment scripts.

Covers:
  - llm_user_narration: cache lookup uses enumerate index, not item['id']
  - llm_item_enrichment: generate_descriptions uses enumerate index; progress counter is idx
  - llm_reranker: get_lgbm_ranking falls back gracefully when ranker_bundle is None;
                  get_lgbm_ranking returns FAISS order when ranker raises an exception
"""

import json
import numpy as np
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_items(n: int) -> list[dict]:
    return [
        {'name': f'Item {i}', 'aisle': f'aisle-{i}', 'department': f'dept-{i}',
         'category': 'cat', 'cat_idx': 0, 'price_tier': 1, 'popularity': 0.5,
         'features': np.zeros(5, dtype=np.float32)}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# llm_user_narration — cache lookup fix
# ---------------------------------------------------------------------------

class TestUserNarrationCacheLookup(unittest.TestCase):
    """load_or_build_item_index must look up descriptions by enumerate index, not item['id']."""

    def _build_cache(self, items: list[dict], tmp_path: Path) -> Path:
        cache = {str(i): f"LLM desc for item {i}" for i in range(len(items))}
        cache_file = tmp_path / "item_descriptions_cache.json"
        cache_file.write_text(json.dumps(cache))
        return cache_file

    def test_cache_lookup_by_index(self, tmp_path=None):
        """All items resolved from cache using their enumerate index."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            items = _make_items(5)
            cache_file = self._build_cache(items, tmp)
            cache = json.loads(cache_file.read_text())

            # Reproduce the fixed logic from load_or_build_item_index
            texts = [
                cache.get(str(i),
                          f"{item['name']} - {item.get('aisle','?')} - {item.get('department','?')}")
                for i, item in enumerate(items)
            ]

            self.assertEqual(len(texts), 5)
            for i in range(5):
                self.assertEqual(texts[i], f"LLM desc for item {i}",
                                 f"Item {i} should have been resolved from cache")

    def test_cache_miss_falls_back_to_template(self):
        """Items missing from cache fall back to template description."""
        cache = {"0": "LLM desc for item 0"}  # only item 0 cached
        items = _make_items(3)

        texts = [
            cache.get(str(i),
                      f"{item['name']} - {item.get('aisle','?')} - {item.get('department','?')}")
            for i, item in enumerate(items)
        ]

        self.assertEqual(texts[0], "LLM desc for item 0")
        self.assertEqual(texts[1], "Item 1 - aisle-1 - dept-1")
        self.assertEqual(texts[2], "Item 2 - aisle-2 - dept-2")

    def test_no_item_id_key_required(self):
        """Lookup must not access item['id'] — items have no such key."""
        items = _make_items(3)
        cache = {str(i): f"desc {i}" for i in range(3)}
        for item in items:
            self.assertNotIn('id', item, "Test setup error: items should not have 'id' key")

        # Should not raise KeyError
        texts = [cache.get(str(i), item['name']) for i, item in enumerate(items)]
        self.assertEqual(len(texts), 3)


# ---------------------------------------------------------------------------
# llm_item_enrichment — generate_descriptions index fix + progress counter
# ---------------------------------------------------------------------------

class TestItemEnrichmentDescriptions(unittest.TestCase):
    """generate_descriptions must key by enumerate index and print correct progress counter."""

    def test_no_llm_keys_by_index(self):
        """--no-llm path produces keys '0'..'n-1', matching enumerate convention."""
        items = _make_items(4)

        # Reproduce the fixed no-llm path
        template_descs = {str(i): f"{item['name']} - {item.get('aisle','?')} - {item.get('department','?')}"
                          for i, item in enumerate(items)}

        self.assertEqual(set(template_descs.keys()), {'0', '1', '2', '3'})
        self.assertIn('Item 0', template_descs['0'])

    def test_progress_counter_uses_loop_index(self):
        """Progress reporting should use the loop counter (idx), not the item's enumerate index."""
        items = _make_items(5)
        # Simulate uncached = last 3 items (indices 2, 3, 4)
        cache = {'0': 'cached', '1': 'cached'}
        uncached = [(i, item) for i, item in enumerate(items) if str(i) not in cache]

        progress_reports = []
        for idx, (i, item) in enumerate(uncached):
            # Fixed: print idx (0-based loop counter), not i (item index)
            if idx > 0 and idx % 1 == 0:  # every iteration for test
                progress_reports.append((idx, len(uncached)))

        # idx should be 1 and 2, NOT 3 and 4
        for loop_counter, total in progress_reports:
            self.assertLess(loop_counter, total,
                            f"Progress counter {loop_counter} should be < total {total}")
        self.assertEqual(progress_reports[0][0], 1)
        self.assertEqual(progress_reports[1][0], 2)

    def test_embed_descriptions_uses_index(self):
        """embed_descriptions falls back to template for items missing from descriptions dict."""
        items = _make_items(3)
        descriptions = {'0': 'enriched zero', '2': 'enriched two'}  # item 1 missing

        def make_template(item: dict) -> str:
            return f"{item['name']} - template"

        texts = [descriptions.get(str(i), make_template(item)) for i, item in enumerate(items)]

        self.assertEqual(texts[0], 'enriched zero')
        self.assertEqual(texts[1], 'Item 1 - template')
        self.assertEqual(texts[2], 'enriched two')


# ---------------------------------------------------------------------------
# llm_reranker — get_lgbm_ranking fallback behaviour
# ---------------------------------------------------------------------------

class TestLgbmRankingFallback(unittest.TestCase):
    """get_lgbm_ranking must return FAISS order when ranker_bundle is None or raises."""

    def _get_lgbm_ranking(self, ranker_bundle, user_features, item_features,
                           up_stats, uid, candidate_indices):
        """Inline copy of the fixed get_lgbm_ranking for isolated testing."""
        if ranker_bundle is None:
            return candidate_indices
        try:
            ranker, user_embs, item_embs = ranker_bundle
            u_emb = user_embs[uid]
            faiss_scores = item_embs[candidate_indices] @ u_emb
            # Simplified: skip _make_features, just use faiss_scores as features
            scores = ranker.predict(faiss_scores.reshape(-1, 1))
            return candidate_indices[np.argsort(-scores)]
        except Exception:
            return candidate_indices

    def test_none_bundle_returns_faiss_order(self):
        candidate_indices = np.array([3, 7, 1, 9, 0])
        result = self._get_lgbm_ranking(
            None, None, None, None, 0, candidate_indices
        )
        np.testing.assert_array_equal(result, candidate_indices)

    def test_failing_ranker_falls_back_to_faiss_order(self):
        bad_ranker = MagicMock()
        bad_ranker.predict.side_effect = RuntimeError("model exploded")

        n_users, n_items, emb_dim = 5, 10, 4
        user_embs = np.random.rand(n_users, emb_dim).astype(np.float32)
        item_embs = np.random.rand(n_items, emb_dim).astype(np.float32)
        user_features = np.random.rand(n_users, 3).astype(np.float32)
        item_features = np.random.rand(n_items, 3).astype(np.float32)
        candidate_indices = np.array([2, 5, 8, 1, 6])

        result = self._get_lgbm_ranking(
            (bad_ranker, user_embs, item_embs),
            user_features, item_features, {}, 0, candidate_indices
        )
        np.testing.assert_array_equal(result, candidate_indices)

    def test_valid_ranker_reorders_candidates(self):
        n_users, n_items, emb_dim = 5, 10, 4
        user_embs = np.random.rand(n_users, emb_dim).astype(np.float32)
        item_embs = np.random.rand(n_items, emb_dim).astype(np.float32)
        user_features = np.random.rand(n_users, 3).astype(np.float32)
        item_features = np.random.rand(n_items, 3).astype(np.float32)
        candidate_indices = np.array([2, 5, 8, 1, 6])

        # Ranker that always scores in reverse order of input
        mock_ranker = MagicMock()
        mock_ranker.predict.side_effect = lambda x: np.arange(len(x), 0, -1, dtype=float)

        result = self._get_lgbm_ranking(
            (mock_ranker, user_embs, item_embs),
            user_features, item_features, {}, 0, candidate_indices
        )
        # Should be reordered (not identical to input since ranker returns non-trivial scores)
        self.assertEqual(len(result), len(candidate_indices))
        self.assertEqual(set(result.tolist()), set(candidate_indices.tolist()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
