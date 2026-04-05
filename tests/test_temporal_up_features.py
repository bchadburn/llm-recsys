import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_instacart import _compute_temporal_up_features


def _make_op_f(records):
    """Build a minimal op_f DataFrame from a list of dicts."""
    df = pd.DataFrame(records)
    df['days_since_prior_order'] = df['days_since_prior_order'].astype(np.float32)
    df['reordered'] = df['reordered'].astype(np.int8)
    df['order_number'] = df['order_number'].astype(np.int16)
    return df


def test_order_streak_consecutive():
    """User bought item in orders 1,2,3 — streak should be 3."""
    op_f = _make_op_f([
        {'user_idx': 0, 'item_idx': 0, 'order_id': 1, 'order_number': 1, 'days_since_prior_order': np.nan, 'reordered': 0},
        {'user_idx': 0, 'item_idx': 0, 'order_id': 2, 'order_number': 2, 'days_since_prior_order': 7.0,    'reordered': 1},
        {'user_idx': 0, 'item_idx': 0, 'order_id': 3, 'order_number': 3, 'days_since_prior_order': 7.0,    'reordered': 1},
    ])
    result = _compute_temporal_up_features(op_f)
    row = result[(result['user_idx'] == 0) & (result['item_idx'] == 0)].iloc[0]
    assert row['up_order_streak'] == 3


def test_order_streak_with_gap():
    """User bought item in orders 1 and 3 (gap at 2) — streak should be 1."""
    op_f = _make_op_f([
        {'user_idx': 0, 'item_idx': 0, 'order_id': 1, 'order_number': 1, 'days_since_prior_order': np.nan, 'reordered': 0},
        {'user_idx': 0, 'item_idx': 0, 'order_id': 3, 'order_number': 3, 'days_since_prior_order': 14.0,   'reordered': 1},
    ])
    result = _compute_temporal_up_features(op_f)
    row = result[(result['user_idx'] == 0) & (result['item_idx'] == 0)].iloc[0]
    assert row['up_order_streak'] == 1


def test_days_since_last_order_recent():
    """Item last bought at order 3 (user max=3) — days_since_last_order should be 0."""
    op_f = _make_op_f([
        {'user_idx': 0, 'item_idx': 0, 'order_id': 3, 'order_number': 3, 'days_since_prior_order': 7.0, 'reordered': 1},
    ])
    result = _compute_temporal_up_features(op_f)
    row = result[(result['user_idx'] == 0) & (result['item_idx'] == 0)].iloc[0]
    assert row['up_days_since_last_order'] == pytest.approx(0.0, abs=1e-4)


def test_days_since_last_order_old():
    """Item last bought at order 1; user's order 2 was 14 days later (max order=2).
    days_since = 14/30 clipped to [0,1]."""
    op_f = _make_op_f([
        {'user_idx': 0, 'item_idx': 0, 'order_id': 1, 'order_number': 1, 'days_since_prior_order': np.nan, 'reordered': 0},
        {'user_idx': 0, 'item_idx': 1, 'order_id': 2, 'order_number': 2, 'days_since_prior_order': 14.0,   'reordered': 1},
    ])
    result = _compute_temporal_up_features(op_f)
    row = result[(result['user_idx'] == 0) & (result['item_idx'] == 0)].iloc[0]
    expected = min(14.0 / 30.0, 1.0)
    assert row['up_days_since_last_order'] == pytest.approx(expected, abs=1e-4)


def test_order_rate():
    """Item bought in 2 out of 3 user orders — up_order_rate should be 2/3."""
    op_f = _make_op_f([
        {'user_idx': 0, 'item_idx': 0, 'order_id': 1, 'order_number': 1, 'days_since_prior_order': np.nan, 'reordered': 0},
        {'user_idx': 0, 'item_idx': 0, 'order_id': 2, 'order_number': 2, 'days_since_prior_order': 7.0,    'reordered': 1},
        {'user_idx': 0, 'item_idx': 1, 'order_id': 3, 'order_number': 3, 'days_since_prior_order': 7.0,    'reordered': 1},
    ])
    result = _compute_temporal_up_features(op_f)
    row = result[(result['user_idx'] == 0) & (result['item_idx'] == 0)].iloc[0]
    assert row['up_order_rate'] == pytest.approx(2 / 3, abs=1e-4)


def test_orders_since_last_order():
    """Item last bought 2 orders ago (order 1, user max=3) — orders_since should be 2/max_orders."""
    op_f = _make_op_f([
        {'user_idx': 0, 'item_idx': 0, 'order_id': 1, 'order_number': 1, 'days_since_prior_order': np.nan, 'reordered': 0},
        {'user_idx': 0, 'item_idx': 1, 'order_id': 2, 'order_number': 2, 'days_since_prior_order': 7.0,    'reordered': 1},
        {'user_idx': 0, 'item_idx': 1, 'order_id': 3, 'order_number': 3, 'days_since_prior_order': 7.0,    'reordered': 1},
    ])
    result = _compute_temporal_up_features(op_f)
    row = result[(result['user_idx'] == 0) & (result['item_idx'] == 0)].iloc[0]
    # max_orders_global = 3; last_order_num for item0 = 1; orders_since = (3-1)/3
    expected = (3 - 1) / 3
    assert row['up_orders_since_last_order'] == pytest.approx(expected, abs=1e-4)
