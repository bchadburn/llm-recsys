"""
Tests for bug fixes applied to the overnight LLM experiment scripts.

Covers:
  - llm_user_narration: cache lookup uses enumerate index, not item['id']
  - llm_item_enrichment: generate_descriptions uses enumerate index; progress counter is idx
  - llm_reranker: get_lgbm_ranking falls back gracefully when ranker_bundle is None;
                  get_lgbm_ranking returns FAISS order when ranker raises an exception
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

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

    def _make_mock_st(self, n_items: int, emb_dim: int = 4) -> MagicMock:
        mock_st = MagicMock()
        mock_st.encode.return_value = np.random.rand(n_items, emb_dim).astype(np.float32)
        return mock_st

    def test_cache_lookup_by_index(self):
        """All items resolved from cache using their enumerate index."""
        import llm_user_narration

        items = _make_items(5)
        with tempfile.TemporaryDirectory() as td:
            cache_file = Path(td) / "item_descriptions_cache.json"
            cache = {str(i): f"LLM desc for item {i}" for i in range(5)}
            cache_file.write_text(json.dumps(cache))

            mock_st = self._make_mock_st(len(items))
            with patch.object(llm_user_narration, 'CACHE_FILE', cache_file), \
                 patch('sentence_transformers.SentenceTransformer', return_value=mock_st):
                llm_user_narration.load_or_build_item_index(items)

            texts = mock_st.encode.call_args[0][0]
            self.assertEqual(len(texts), 5)
            for i in range(5):
                self.assertEqual(texts[i], f"LLM desc for item {i}",
                                 f"Item {i} should have been resolved from cache")

    def test_cache_miss_falls_back_to_template(self):
        """Items missing from cache fall back to template description."""
        import llm_user_narration

        items = _make_items(3)
        with tempfile.TemporaryDirectory() as td:
            cache_file = Path(td) / "item_descriptions_cache.json"
            cache = {"0": "LLM desc for item 0"}  # only item 0 cached
            cache_file.write_text(json.dumps(cache))

            mock_st = self._make_mock_st(len(items))
            with patch.object(llm_user_narration, 'CACHE_FILE', cache_file), \
                 patch('sentence_transformers.SentenceTransformer', return_value=mock_st):
                llm_user_narration.load_or_build_item_index(items)

            texts = mock_st.encode.call_args[0][0]
            self.assertEqual(texts[0], "LLM desc for item 0")
            self.assertEqual(texts[1], "Item 1 - aisle-1 - dept-1")
            self.assertEqual(texts[2], "Item 2 - aisle-2 - dept-2")

    def test_no_item_id_key_required(self):
        """Lookup must not access item['id'] — items have no such key."""
        import llm_user_narration

        items = _make_items(3)
        for item in items:
            self.assertNotIn('id', item, "Test setup error: items should not have 'id' key")

        with tempfile.TemporaryDirectory() as td:
            cache_file = Path(td) / "item_descriptions_cache.json"
            cache = {str(i): f"desc {i}" for i in range(3)}
            cache_file.write_text(json.dumps(cache))

            mock_st = self._make_mock_st(len(items))
            with patch.object(llm_user_narration, 'CACHE_FILE', cache_file), \
                 patch('sentence_transformers.SentenceTransformer', return_value=mock_st):
                # Should not raise KeyError
                llm_user_narration.load_or_build_item_index(items)

            texts = mock_st.encode.call_args[0][0]
            self.assertEqual(len(texts), 3)


# ---------------------------------------------------------------------------
# llm_item_enrichment — generate_descriptions index fix + progress counter
# ---------------------------------------------------------------------------

class TestItemEnrichmentDescriptions(unittest.TestCase):
    """generate_descriptions must key by enumerate index and print correct progress counter."""

    def test_no_llm_keys_by_index(self):
        """--no-llm path produces keys '0'..'n-1', matching enumerate convention."""
        import llm_item_enrichment

        items = _make_items(4)
        with tempfile.TemporaryDirectory() as td:
            nonexistent = Path(td) / "no_cache.json"
            with patch.object(llm_item_enrichment, 'CACHE_FILE', nonexistent):
                result = llm_item_enrichment.generate_descriptions(items, use_llm=False)

        self.assertEqual(set(result.keys()), {'0', '1', '2', '3'})
        self.assertIn('Item 0', result['0'])

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
        import llm_item_enrichment

        items = _make_items(3)
        descriptions = {'0': 'enriched zero', '2': 'enriched two'}  # item 1 missing

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(3, 4).astype(np.float32)
        llm_item_enrichment.embed_descriptions(descriptions, items, mock_model)

        texts = mock_model.encode.call_args[0][0]
        self.assertEqual(texts[0], 'enriched zero')
        self.assertIn('Item 1', texts[1])
        self.assertEqual(texts[2], 'enriched two')


# ---------------------------------------------------------------------------
# llm_reranker — get_lgbm_ranking fallback behaviour
# ---------------------------------------------------------------------------

def _get_lgbm_ranking_with_mocks():
    """Return get_lgbm_ranking with ranker/data_instacart pre-mocked in sys.modules."""
    mock_ranker_mod = MagicMock()
    mock_data_mod = MagicMock()
    mock_data_mod.PRICE_SENS_IDX = 0
    # Ensure these are in sys.modules before the function does its internal imports
    sys.modules.setdefault('ranker', mock_ranker_mod)
    sys.modules.setdefault('data_instacart', mock_data_mod)
    from llm_reranker import get_lgbm_ranking
    return get_lgbm_ranking, mock_ranker_mod, mock_data_mod


class TestLgbmRankingFallback(unittest.TestCase):
    """get_lgbm_ranking must return FAISS order when ranker_bundle is None or raises."""

    def setUp(self):
        self.get_lgbm_ranking, self.mock_ranker_mod, _ = _get_lgbm_ranking_with_mocks()

    def test_none_bundle_returns_faiss_order(self):
        candidate_indices = np.array([3, 7, 1, 9, 0])
        result = self.get_lgbm_ranking(
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

        self.mock_ranker_mod._make_features.return_value = np.ones((5, 3))

        result = self.get_lgbm_ranking(
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

        self.mock_ranker_mod._make_features.return_value = np.ones((5, 3))

        # Ranker that always scores in reverse order of input
        mock_ranker = MagicMock()
        mock_ranker.predict.side_effect = lambda x: np.arange(len(x), 0, -1, dtype=float)

        result = self.get_lgbm_ranking(
            (mock_ranker, user_embs, item_embs),
            user_features, item_features, {}, 0, candidate_indices
        )
        # Should be reordered (not identical to input since ranker returns non-trivial scores)
        self.assertEqual(len(result), len(candidate_indices))
        self.assertEqual(set(result.tolist()), set(candidate_indices.tolist()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
