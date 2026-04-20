"""
data_instacart.py — Instacart Market Basket Analysis data loader.

Provides the same return interface as generate_data() in data.py so all
downstream code (main.py, ranker.py, eval.py) works without modification.

Expected files in data_dir (download from Kaggle):
    departments.csv
    aisles.csv
    products.csv
    orders.csv
    order_products__prior.csv

Download:
    # Option A — Kaggle CLI (requires ~/.kaggle/kaggle.json)
    kaggle competitions download -c instacart-market-basket-analysis
    unzip instacart-market-basket-analysis.zip -d data/

    # Option B — Manual
    https://www.kaggle.com/competitions/instacart-market-basket-analysis/data
    Place the unzipped CSVs into a local directory, e.g. data/instacart/

Run main.py with the data:
    python main.py --data-dir data/instacart/
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ── Department taxonomy ────────────────────────────────────────────────────────
# Instacart has 21 departments (department_id 1–21 in the CSV).
# Index here = department_id - 1  (zero-based).

DEPARTMENTS = [
    'frozen',           # 1
    'other',            # 2
    'bakery',           # 3
    'produce',          # 4
    'alcohol',          # 5
    'international',    # 6
    'beverages',        # 7
    'pets',             # 8
    'dry goods pasta',  # 9
    'bulk',             # 10
    'personal care',    # 11
    'meat seafood',     # 12
    'pantry',           # 13
    'breakfast',        # 14
    'canned goods',     # 15
    'dairy eggs',       # 16
    'household',        # 17
    'babies',           # 18
    'snacks',           # 19
    'deli',             # 20
    'missing',          # 21
]

N_DEPTS = len(DEPARTMENTS)
_DEPT_IDX = {d: i for i, d in enumerate(DEPARTMENTS)}

# Labels used for the demo display in main.py (matching DEPARTMENTS entries)
ARCHETYPE_LABELS = ['produce', 'snacks', 'household']

# Price tier heuristic — Instacart has no price column, so we assign based
# on department spending patterns. 0=budget, 1=mid, 2=premium.
_DEPT_PRICE_TIER = {
    'frozen': 0, 'dry goods pasta': 0, 'bulk': 0, 'pantry': 0,
    'breakfast': 0, 'canned goods': 0, 'household': 0, 'snacks': 0,
    'beverages': 0,
    'produce': 1, 'bakery': 1, 'dairy eggs': 1, 'international': 1,
    'personal care': 1, 'pets': 1, 'other': 1, 'missing': 1,
    'alcohol': 2, 'meat seafood': 2, 'babies': 2, 'deli': 2,
}

# Departments that signal willingness to spend more (used as price-sensitivity proxy)
_PREMIUM_DEPTS = {'alcohol', 'meat seafood', 'babies', 'deli'}

# ── Feature layout constants (imported by ranker.py and main.py) ───────────────
#
# User feature vector (49d):
#   [0:21]  dept_log_counts  — log1p-normalized purchase count per department
#   [21:42] dept_prefs       — fraction of purchases per department (sums to 1)
#   [42]    total_orders     — normalized total purchase count
#   [43]    recency          — normalized avg days between orders (higher = more recent)
#   [44]    reorder_rate     — fraction of purchased items that are re-purchases
#   [45]    variety          — normalized Shannon entropy of dept distribution
#   [46]    premium_share    — share of purchases from premium departments
#   [47]    avg_basket_size  — normalized avg items per order
#   [48]    order_gap_std    — normalized std of days between orders (regularity signal)
#
# Item feature vector (24d):
#   [0:21]  dept_onehot      — one-hot department
#   [21]    popularity       — log1p-normalized purchase count
#   [22]    reorder_rate     — fraction of purchases that are re-purchases
#   [23]    avg_cart_pos     — normalized mean add-to-cart position
#
# User×Product (UP) interaction features (7d):
#   [0]     up_purchase_count_norm  — log1p-normalised count of times user bought item
#   [1]     up_reorder_rate         — fraction of those purchases that are reorders
#   [2]     up_user_order_frac      — purchase_count / user's total interactions
#   [3]     up_days_since_last_order    — days since last purchase, /30 clipped [0,1]
#   [4]     up_orders_since_last_order  — orders since last purchase, norm by global max
#   [5]     up_order_streak_norm        — consecutive recent orders streak, norm by max
#   [6]     up_order_rate               — orders_with_item / total_user_orders

USER_FEATURE_DIM = 49
ITEM_FEATURE_DIM = 24
PRICE_SENS_IDX   = 46   # index of premium_share in user feature vector
POPULARITY_IDX   = 21   # index of popularity in item feature vector
USER_PREFS_SLICE = slice(21, 42)   # dept preference fractions


# ── File validation ────────────────────────────────────────────────────────────

def _check_files(data_dir: Path) -> None:
    required = [
        'departments.csv', 'aisles.csv', 'products.csv',
        'orders.csv', 'order_products__prior.csv',
    ]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Instacart files in {data_dir}: {missing}\n\n"
            "Download from Kaggle:\n"
            "  kaggle competitions download -c instacart-market-basket-analysis\n"
            "  unzip instacart-market-basket-analysis.zip -d <data_dir>\n\n"
            "Or manually: https://www.kaggle.com/competitions/instacart-market-basket-analysis/data"
        )


# ── Temporal user-product features ────────────────────────────────────────────

def _compute_temporal_up_features(op_f: pd.DataFrame) -> pd.DataFrame:
    """
    Compute temporal user-product features from interaction data.

    Requires op_f to have columns: user_idx, item_idx, order_id,
    order_number, days_since_prior_order.

    Returns a DataFrame with columns:
        user_idx, item_idx,
        up_days_since_last_order,    — normalized to [0,1] by /30, clipped
        up_orders_since_last_order,  — normalized by global max order count
        up_order_streak,             — raw consecutive order count (normalize in caller)
        up_order_rate                — orders_with_item / total_user_orders
    """
    # ── Order-level timeline (one row per order, not per item) ────────────────
    order_tl = (
        op_f[['user_idx', 'order_id', 'order_number', 'days_since_prior_order']]
        .drop_duplicates('order_id')
        .sort_values(['user_idx', 'order_number'])
        .copy()
    )
    order_tl['days_since_prior_order'] = order_tl['days_since_prior_order'].fillna(0.0)
    order_tl['cum_days'] = order_tl.groupby('user_idx')['days_since_prior_order'].cumsum()

    user_max_cum_days  = order_tl.groupby('user_idx')['cum_days'].max()
    user_max_order_num = order_tl.groupby('user_idx')['order_number'].max()

    # ── Last order per (user, item) ───────────────────────────────────────────
    up_last = (
        op_f.groupby(['user_idx', 'item_idx'])['order_number']
        .max()
        .reset_index()
        .rename(columns={'order_number': 'last_order_num'})
    )
    # Join cumulative days at the last purchase order
    up_last = up_last.merge(
        order_tl[['user_idx', 'order_number', 'cum_days']].rename(
            columns={'order_number': 'last_order_num'}
        ),
        on=['user_idx', 'last_order_num'],
        how='left',
    )
    up_last = up_last.merge(user_max_cum_days.rename('max_cum_days'), on='user_idx')
    up_last = up_last.merge(user_max_order_num.rename('max_order_num'), on='user_idx')

    max_orders_global = float(up_last['max_order_num'].max() or 1)

    up_last['up_days_since_last_order'] = (
        (up_last['max_cum_days'] - up_last['cum_days']).clip(lower=0) / 30.0
    ).clip(upper=1.0).astype(np.float32)

    up_last['up_orders_since_last_order'] = (
        (up_last['max_order_num'] - up_last['last_order_num']) / max_orders_global
    ).clip(0, 1).astype(np.float32)

    # ── Order streak ──────────────────────────────────────────────────────────
    # Streak = length of the consecutive tail of orders containing this item.
    # e.g. orders [1,2,3] → streak 3; orders [1,3] → streak 1 (gap at 2).
    ui_orders = op_f[['user_idx', 'item_idx', 'order_number']].drop_duplicates()
    ui_orders = ui_orders.sort_values(['user_idx', 'item_idx', 'order_number'])
    ui_orders['max_ui_order']  = ui_orders.groupby(['user_idx', 'item_idx'])['order_number'].transform('max')
    ui_orders['gap_from_max']  = ui_orders['max_ui_order'] - ui_orders['order_number']
    ui_orders['rank_from_end'] = ui_orders.groupby(['user_idx', 'item_idx']).cumcount(ascending=False)
    ui_orders['in_streak']     = (ui_orders['gap_from_max'] == ui_orders['rank_from_end']).astype(np.int32)

    up_streak = (
        ui_orders[ui_orders['in_streak'] == 1]
        .groupby(['user_idx', 'item_idx'])
        .size()
        .reset_index(name='up_order_streak')
    )

    # ── Order rate ────────────────────────────────────────────────────────────
    # Fraction of user's orders that contain this item (uses order count not interaction count).
    user_total_orders = op_f.groupby('user_idx')['order_number'].nunique()
    up_order_rate = (
        op_f.groupby(['user_idx', 'item_idx'])['order_number']
        .nunique()
        .reset_index(name='orders_with_item')
    )
    up_order_rate = up_order_rate.merge(user_total_orders.rename('total_user_orders'), on='user_idx')
    up_order_rate['up_order_rate'] = (
        up_order_rate['orders_with_item'] / up_order_rate['total_user_orders']
    ).astype(np.float32)

    # ── Merge into a single DataFrame ─────────────────────────────────────────
    result = up_last[['user_idx', 'item_idx', 'up_days_since_last_order', 'up_orders_since_last_order']].copy()
    result = result.merge(up_streak[['user_idx', 'item_idx', 'up_order_streak']], on=['user_idx', 'item_idx'], how='left')
    result = result.merge(up_order_rate[['user_idx', 'item_idx', 'up_order_rate']], on=['user_idx', 'item_idx'], how='left')
    # fillna(0) is a safety guard; every (user, item) pair always has at least
    # a streak of 1 (the last purchase always satisfies gap_from_max == 0).
    result['up_order_streak'] = result['up_order_streak'].fillna(0).astype(np.int32)
    result['up_order_rate']   = result['up_order_rate'].fillna(0.0)
    return result


# ── Main loader ────────────────────────────────────────────────────────────────

def load_instacart(
    data_dir:        str | Path,
    n_users:         int = 5000,
    n_items:         int = 10000,
    n_interactions:  int = 200_000,
    seed:            int = 42,
) -> tuple:
    """
    Load Instacart Market Basket Analysis data into the standard interface.

    Subsamples to the top-n_items products by purchase count and the top-n_users
    users by order count within that item set. User and item features are built
    from the full interaction history; the interactions list returned for training
    is capped at n_interactions (randomly sampled) to keep epoch time tractable.

    Returns:
        user_features:   np.ndarray [n_users, 47]
        item_features:   np.ndarray [n_items, 24]
        items:           list[dict] with keys: name, category, aisle, cat_idx,
                         price_tier, popularity, features
        interactions:    list[(user_idx, item_idx)]
        user_archetypes: np.ndarray [n_users] — dominant dept index per user
        up_stats:        dict[(user_idx, item_idx), np.ndarray[7]] — UP features:
                         [up_purchase_count_norm, up_reorder_rate, up_user_order_frac,
                          up_days_since_last_order, up_orders_since_last_order,
                          up_order_streak_norm, up_order_rate]
    """
    data_dir = Path(data_dir)
    _check_files(data_dir)

    # ── Parquet cache ─────────────────────────────────────────────────────────
    # order_products__prior.csv has ~32M rows; reading + filtering it takes ~50s.
    # Cache the filtered slice to parquet so subsequent runs load in <2s.
    cache_path = data_dir / f"_cache_u{n_users}_i{n_items}_v2.parquet"
    if cache_path.exists():
        print(f"  Loading from cache: {cache_path.name}", flush=True)
        op_f = pd.read_parquet(cache_path)
        n_i  = op_f['item_idx'].max() + 1
        n_u  = op_f['user_idx'].max() + 1
        # Re-load lightweight product metadata for building items list
        depts    = pd.read_csv(data_dir / 'departments.csv')
        aisles   = pd.read_csv(data_dir / 'aisles.csv')
        products = pd.read_csv(data_dir / 'products.csv')
        products = products.merge(aisles, on='aisle_id').merge(depts, on='department_id')
        products['department'] = products['department'].str.strip().str.lower()
        products['aisle']      = products['aisle'].str.strip().str.lower()
        pids_used = (
            op_f[['product_id', 'item_idx']]
            .drop_duplicates('item_idx')
            .sort_values('item_idx')['product_id']
            .tolist()
        )
        print(f"  {n_u:,} users  |  {n_i:,} items  |  {len(op_f):,} interactions", flush=True)
    else:
        print("  Loading CSV files (first run — will cache for next time)...", flush=True)
        depts    = pd.read_csv(data_dir / 'departments.csv')
        aisles   = pd.read_csv(data_dir / 'aisles.csv')
        products = pd.read_csv(data_dir / 'products.csv')
        orders   = pd.read_csv(data_dir / 'orders.csv')
        op       = pd.read_csv(data_dir / 'order_products__prior.csv')

        prior_orders = orders[orders['eval_set'] == 'prior'].copy()

        products = (
            products
            .merge(aisles, on='aisle_id')
            .merge(depts,  on='department_id')
        )
        products['department'] = products['department'].str.strip().str.lower()
        products['aisle']      = products['aisle'].str.strip().str.lower()

        print(f"  Filtering to top {n_items:,} items and top {n_users:,} users...", flush=True)
        item_counts = op.groupby('product_id').size()
        top_pids    = set(item_counts.nlargest(n_items).index)
        op_f        = op[op['product_id'].isin(top_pids)].copy()

        valid_order_ids = set(op_f['order_id'].unique())
        prior_valid     = prior_orders[prior_orders['order_id'].isin(valid_order_ids)]
        user_order_n    = prior_valid.groupby('user_id')['order_id'].nunique()
        top_uids        = set(user_order_n.nlargest(n_users).index)

        prior_top = prior_orders[prior_orders['user_id'].isin(top_uids)].copy()
        op_f      = op_f[op_f['order_id'].isin(prior_top['order_id'])].copy()

        op_f = op_f.merge(
            prior_top[['order_id', 'user_id', 'days_since_prior_order', 'order_number']],
            on='order_id',
        )
        op_f = op_f.merge(
            products[['product_id', 'product_name', 'aisle', 'department']], on='product_id'
        )

        pids_used = sorted(op_f['product_id'].unique())
        uids_used = sorted(top_uids & set(op_f['user_id'].unique()))
        n_i = len(pids_used)
        n_u = len(uids_used)

        pid_to_idx = {p: i for i, p in enumerate(pids_used)}
        uid_to_idx = {u: i for i, u in enumerate(uids_used)}

        op_f['item_idx'] = op_f['product_id'].map(pid_to_idx)
        op_f['user_idx'] = op_f['user_id'].map(uid_to_idx)
        op_f = op_f.dropna(subset=['item_idx', 'user_idx'])
        op_f[['item_idx', 'user_idx']] = op_f[['item_idx', 'user_idx']].astype(int)

        # Downcast numeric columns to reduce memory ~5x on the ~32M-row prior file.
        # int32 covers item/user indices (max ~200k); float32 covers cart pos and days.
        op_f['item_idx']               = op_f['item_idx'].astype(np.int32)
        op_f['user_idx']               = op_f['user_idx'].astype(np.int32)
        op_f['order_id']               = op_f['order_id'].astype(np.int32)
        op_f['product_id']             = op_f['product_id'].astype(np.int32)
        op_f['reordered']              = op_f['reordered'].astype(np.int8)
        op_f['add_to_cart_order']      = op_f['add_to_cart_order'].astype(np.int16)
        op_f['days_since_prior_order'] = op_f['days_since_prior_order'].astype(np.float32)
        op_f['order_number']           = op_f['order_number'].astype(np.int16)

        print(f"  {n_u:,} users  |  {n_i:,} items  |  {len(op_f):,} interactions", flush=True)
        print(f"  Saving cache to {cache_path.name}...", flush=True)
        op_f.to_parquet(cache_path, index=False)
        print("  Cache saved.", flush=True)

    # ── Item features ─────────────────────────────────────────────────────────
    item_stats = (
        op_f.groupby('item_idx')
        .agg(
            purchase_count  = ('item_idx',          'size'),
            reorder_sum     = ('reordered',          'sum'),
            avg_cart_pos    = ('add_to_cart_order',  'mean'),
        )
        .reindex(range(n_i), fill_value=0)
    )
    max_purchases = float(item_stats['purchase_count'].max() or 1)
    max_cart_pos  = float(item_stats['avg_cart_pos'].max() or 1)

    prod_info = products.set_index('product_id')
    items = []
    item_features_list = []

    for local_idx, pid in enumerate(pids_used):
        row  = prod_info.loc[pid]
        dept = row['department'] if row['department'] in _DEPT_IDX else 'missing'
        aisle, name = row['aisle'], row['product_name']

        dept_idx    = _DEPT_IDX[dept]
        dept_onehot = np.zeros(N_DEPTS, dtype=np.float32)
        dept_onehot[dept_idx] = 1.0

        s            = item_stats.loc[local_idx]
        popularity   = float(np.log1p(s['purchase_count']) / np.log1p(max_purchases))
        reorder_rate = float(s['reorder_sum'] / s['purchase_count']) if s['purchase_count'] > 0 else 0.0
        cart_pos     = float(s['avg_cart_pos'] / max_cart_pos)

        feat = np.concatenate(
            [dept_onehot, [popularity, reorder_rate, cart_pos]]
        ).astype(np.float32)

        items.append({
            'name':       name,
            'category':   dept,           # 'category' key for compatibility with main.py
            'aisle':      aisle,
            'cat_idx':    dept_idx,
            'price_tier': _DEPT_PRICE_TIER.get(dept, 1),
            'popularity': popularity,     # pre-extracted for ranker.py
            'features':   feat,
        })
        item_features_list.append(feat)

    item_features = np.array(item_features_list)

    # ── User features (numpy scatter — avoids pivot_table on 1.4M rows) ─────────
    # Map department strings to indices using a dict (vectorized, no Python loop).
    _dept_map = {
        d: _DEPT_IDX.get(d, _DEPT_IDX['missing'])
        for d in op_f['department'].unique()
    }
    dept_idx_arr = op_f['department'].map(_dept_map).values.astype(np.int32)
    user_idx_arr = op_f['user_idx'].values.astype(np.int32)
    reordered_arr = op_f['reordered'].values.astype(np.float32)
    days_arr = op_f['days_since_prior_order'].values.astype(np.float64)

    # Per-user department purchase counts  [n_u, 21]
    dept_counts = np.zeros((n_u, N_DEPTS), dtype=np.float32)
    np.add.at(dept_counts, (user_idx_arr, dept_idx_arr), 1)

    # Per-user reorder counts
    reorder_counts = np.zeros(n_u, dtype=np.float32)
    np.add.at(reorder_counts, user_idx_arr, reordered_arr)

    # Per-user mean days since prior order (recency signal)
    recency_sum = np.zeros(n_u, dtype=np.float64)
    recency_n   = np.zeros(n_u, dtype=np.float64)
    valid_mask  = ~np.isnan(days_arr)
    np.add.at(recency_sum, user_idx_arr[valid_mask], days_arr[valid_mask])
    np.add.at(recency_n,   user_idx_arr[valid_mask], 1.0)
    recency_mean = np.where(recency_n > 0, recency_sum / recency_n, 30.0).astype(np.float32)

    totals = dept_counts.sum(axis=1, keepdims=True)                   # [n_u, 1]
    prefs  = dept_counts / (totals + 1e-8)                            # [n_u, 21]

    log_counts  = np.log1p(dept_counts)
    norm_counts = log_counts / (log_counts.max(axis=1, keepdims=True) + 1e-8)

    max_total  = float(totals.max() or 1)
    total_norm = np.clip(np.log1p(totals.ravel()) / np.log1p(max_total + 1e-8), 0, 1)
    recency_norm = np.clip(1.0 - recency_mean / 30.0, 0, 1)
    reorder_rate = reorder_counts / (totals.ravel() + 1e-8)

    p       = prefs + 1e-10
    entropy = -(p * np.log(p)).sum(axis=1)
    variety = entropy / np.log(N_DEPTS)

    premium_cols  = [_DEPT_IDX[d] for d in _PREMIUM_DEPTS if d in _DEPT_IDX]
    premium_share = dept_counts[:, premium_cols].sum(axis=1) / (totals.ravel() + 1e-8)

    # Avg basket size: mean items-per-order per user
    basket_sizes   = (
        op_f.groupby(['user_idx', 'order_id']).size()
        .groupby(level='user_idx').mean()
        .reindex(range(n_u), fill_value=1.0)
        .values.astype(np.float32)
    )
    max_basket = float(basket_sizes.max() or 1)
    basket_norm = np.clip(basket_sizes / max_basket, 0, 1)

    # Order gap std: predictability of purchase cadence
    order_gap_std_vals = (
        op_f.dropna(subset=['days_since_prior_order'])
        .groupby('user_idx')['days_since_prior_order']
        .std()
        .reindex(range(n_u), fill_value=0.0)
        .fillna(0.0)
        .values.astype(np.float32)
    )
    max_gap_std = float(order_gap_std_vals.max() or 1)
    gap_std_norm = np.clip(order_gap_std_vals / max_gap_std, 0, 1)

    user_features = np.hstack([
        norm_counts,                        # 21
        prefs,                              # 21
        total_norm.reshape(-1, 1),          # 1
        recency_norm.reshape(-1, 1),        # 1
        reorder_rate.reshape(-1, 1),        # 1
        variety.reshape(-1, 1),             # 1
        premium_share.reshape(-1, 1),       # 1  ← index PRICE_SENS_IDX = 46
        basket_norm.reshape(-1, 1),         # 1
        gap_std_norm.reshape(-1, 1),        # 1
    ]).astype(np.float32)

    user_archetypes = np.argmax(dept_counts, axis=1)

    # ── User×Product (UP) interaction features ────────────────────────────────
    # These are the single most predictive feature category per Kaggle competition
    # analysis. Computed from the full interaction history (not capped).
    #
    # Features per (user_idx, item_idx) pair:
    #   [0] up_purchase_count_norm  — log1p-normalised count of times user bought item
    #   [1] up_reorder_rate         — fraction of those purchases that are reorders
    #   [2] up_user_order_frac      — purchase_count / user's total interactions
    #   [3] up_days_since_last_order    — days since last purchase, /30 clipped [0,1]
    #   [4] up_orders_since_last_order  — orders since last purchase, norm by global max
    #   [5] up_order_streak_norm        — consecutive recent orders streak, norm by max
    #   [6] up_order_rate               — orders_with_item / total_user_orders
    #
    # Stored as a dict for O(1) lookup in ranker._make_features().
    up_grp = (
        op_f
        .groupby(['user_idx', 'item_idx'], sort=False)
        .agg(up_count=('item_idx', 'size'), up_reorder=('reordered', 'sum'))
        .reset_index()
    )
    user_interaction_totals = op_f.groupby('user_idx').size()

    max_up_count = float(up_grp['up_count'].max() or 1)
    log_max      = float(np.log1p(max_up_count))

    temporal_up = _compute_temporal_up_features(op_f)
    max_streak = float(temporal_up['up_order_streak'].max() or 1)
    temporal_up['up_order_streak_norm'] = (temporal_up['up_order_streak'] / max_streak).astype(np.float32)

    up_grp = up_grp.merge(
        temporal_up[['user_idx', 'item_idx',
                     'up_days_since_last_order', 'up_orders_since_last_order',
                     'up_order_streak_norm', 'up_order_rate']],
        on=['user_idx', 'item_idx'],
        how='left',
    )
    up_grp['up_days_since_last_order']   = up_grp['up_days_since_last_order'].fillna(1.0)
    up_grp['up_orders_since_last_order'] = up_grp['up_orders_since_last_order'].fillna(1.0)
    up_grp['up_order_streak_norm']       = up_grp['up_order_streak_norm'].fillna(0.0)
    up_grp['up_order_rate']              = up_grp['up_order_rate'].fillna(0.0)

    up_stats: dict[tuple[int, int], np.ndarray] = {}
    for row in up_grp.itertuples(index=False):
        u   = int(row.user_idx)
        i   = int(row.item_idx)
        cnt = int(row.up_count)
        rer = int(row.up_reorder)
        user_total = float(user_interaction_totals.get(u, 1))
        up_stats[(u, i)] = np.array([
            float(np.log1p(cnt)) / log_max,              # [0] up_purchase_count_norm
            float(rer) / cnt if cnt > 0 else 0.0,         # [1] up_reorder_rate
            float(cnt) / user_total,                       # [2] up_user_order_frac
            float(row.up_days_since_last_order),           # [3] up_days_since_last_order
            float(row.up_orders_since_last_order),         # [4] up_orders_since_last_order
            float(row.up_order_streak_norm),               # [5] up_order_streak_norm
            float(row.up_order_rate),                      # [6] up_order_rate
        ], dtype=np.float32)

    print(f"  UP interaction pairs: {len(up_stats):,}", flush=True)

    # ── Interactions list ─────────────────────────────────────────────────────
    # User/item features above are built from the full history.
    # Cap the training interaction list so each epoch stays tractable on CPU.
    op_train = op_f
    if n_interactions is not None and len(op_f) > n_interactions:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(op_f), size=n_interactions, replace=False)
        op_train = op_f.iloc[np.sort(idx)]
        print(f"  Sampled {n_interactions:,} training interactions from {len(op_f):,}", flush=True)

    interactions = list(zip(
        op_train['user_idx'].tolist(),
        op_train['item_idx'].tolist(),
    ))

    return user_features, item_features, items, interactions, user_archetypes, up_stats


# ── Text embeddings ────────────────────────────────────────────────────────────

def get_item_text_embeddings(items: list) -> np.ndarray:
    """
    Embed each Instacart item as a sentence using all-MiniLM-L6-v2 (384d).

    Template: "{name} - {aisle} - {category}"
    e.g.: "Organic Whole Milk - milk - dairy eggs"

    Using aisle (134 unique values) rather than department (21 values) gives
    much finer-grained semantic distinctions within a department — "fresh fruits"
    vs "fresh vegetables" are both produce but map to clearly different positions
    in embedding space.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [
        f"{it['name']} - {it['aisle']} - {it['category']}"
        for it in items
    ]
    embeddings = model.encode(
        texts, convert_to_numpy=True, show_progress_bar=True, batch_size=256
    )
    return embeddings.astype(np.float32)
