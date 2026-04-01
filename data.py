import numpy as np
import torch
from torch.utils.data import Dataset

CATEGORIES = ['produce', 'dairy', 'meat', 'bakery', 'beverages', 'snacks', 'frozen', 'cleaning']
PRICE_TIERS = ['budget', 'mid', 'premium']
N_CATS = len(CATEGORIES)

# 25 items per category = 200 items total
ITEM_NAMES = {
    'produce': [
        'Organic Spinach', 'Bananas', 'Roma Tomatoes', 'Broccoli', 'Avocado',
        'Blueberries', 'Carrots', 'Kale', 'Red Peppers', 'Garlic',
        'Sweet Potatoes', 'Apples', 'Oranges', 'Cucumber', 'Lemons',
        'Grapes', 'Zucchini', 'Onions', 'Strawberries', 'Celery',
        'Mushrooms', 'Pineapple', 'Mango', 'Watermelon', 'Green Beans',
    ],
    'dairy': [
        'Whole Milk', 'Greek Yogurt', 'Cheddar Cheese', 'Butter', 'Cream Cheese',
        'Heavy Cream', 'Sour Cream', 'Mozzarella', 'Parmesan', 'Almond Milk',
        'Oat Milk', 'Eggs (12-pack)', 'Gouda', 'Brie', 'Feta Cheese',
        'Cottage Cheese', 'Whipping Cream', 'Swiss Cheese', 'Provolone', 'Ricotta',
        'Colby Jack', 'Half and Half', 'Kefir', 'Cream Cheese Spread', 'Eggnog',
    ],
    'meat': [
        'Chicken Breast', 'Ground Beef', 'Salmon Fillet', 'Bacon', 'Pork Chops',
        'Turkey Breast', 'Ribeye Steak', 'Shrimp', 'Lamb Chops', 'Tuna Steak',
        'Italian Sausage', 'Chicken Thighs', 'Beef Tenderloin', 'Crab Legs', 'Tilapia',
        'Ground Turkey', 'Pork Belly', 'Sirloin Steak', 'Cod Fillet', 'Duck Breast',
        'Chicken Wings', 'Bison Burger', 'Lobster Tail', 'Catfish', 'Ham Steak',
    ],
    'bakery': [
        'Sourdough Bread', 'Bagels', 'Croissants', 'Whole Wheat Bread', 'Baguette',
        'Blueberry Muffins', 'Cinnamon Rolls', 'Rye Bread', 'Dinner Rolls', 'Focaccia',
        'Chocolate Chip Cookies', 'Pita Bread', 'English Muffins', 'Brioche', 'Pretzel Rolls',
        'Ciabatta', 'Multigrain Bread', 'Banana Bread', 'Scones', 'Brownies',
        'Kaiser Rolls', 'Marble Cake', 'Apple Pie', 'Biscotti', 'Pumpernickel',
    ],
    'beverages': [
        'Orange Juice', 'Sparkling Water', 'Coffee Beans', 'Green Tea', 'Coca-Cola',
        'Almond Latte', 'Red Wine', 'Craft Beer', 'Kombucha', 'Gatorade',
        'Cold Brew Coffee', 'Lemonade', 'Apple Juice', 'White Wine', 'Iced Tea',
        'Sports Drink', 'Coconut Water', 'Herbal Tea', 'Seltzer Water', 'Protein Shake',
        'Energy Drink', 'Grape Juice', 'Cranberry Juice', 'Espresso Pods', 'Chai Tea',
    ],
    'snacks': [
        'Mixed Nuts', 'Potato Chips', 'Granola Bars', 'Popcorn', 'Dark Chocolate',
        'Trail Mix', 'Pretzels', 'Rice Cakes', 'Beef Jerky', 'Hummus',
        'Cheese Crackers', 'Veggie Straws', 'Almond Butter', 'Peanut Butter Cups', 'Crackers',
        'Dried Mango', 'Gummy Bears', 'Protein Bars', 'Seaweed Snacks', 'Sunflower Seeds',
        'Tortilla Chips', 'Peanuts', 'Chocolate Almonds', 'Fruit Roll-Ups', 'Granola',
    ],
    'frozen': [
        'Frozen Pizza', 'Ice Cream', 'Edamame', 'Fish Sticks', 'Frozen Burritos',
        'Chicken Nuggets', 'Frozen Waffles', 'Peas and Carrots', 'Frozen Lasagna', 'Sorbet',
        'Veggie Burgers', 'Frozen Shrimp', 'Breakfast Sandwiches', 'Tater Tots', 'Frozen Berries',
        'Pot Pies', 'Mozzarella Sticks', 'Frozen Stir Fry', 'Ice Cream Bars', 'Frozen Ravioli',
        'Hash Browns', 'Frozen Tamales', 'Gelato', 'Frozen Green Beans', 'Chimichanga',
    ],
    'cleaning': [
        'Dish Soap', 'Laundry Detergent', 'All-Purpose Cleaner', 'Paper Towels', 'Sponges',
        'Bleach', 'Fabric Softener', 'Trash Bags', 'Glass Cleaner', 'Disinfectant Wipes',
        'Toilet Bowl Cleaner', 'Mop Refills', 'Dryer Sheets', 'Scrub Brushes', 'Baking Soda',
        'Vinegar', 'Air Freshener', 'Rubber Gloves', 'Steel Wool Pads', 'Furniture Polish',
        'Floor Cleaner', 'Stain Remover', 'Cleaning Spray', 'Dish Pods', 'Lint Roller',
    ],
}


def generate_data(n_users=500, n_items=200, n_interactions=5000, seed=42):
    """
    Returns:
        user_features:    np.ndarray [n_users, 20]
        item_features:    np.ndarray [n_items, 13]
        items:            list of dicts with name/category/price_tier metadata
        interactions:     list of (user_id, item_id) tuples
        user_archetypes:  np.ndarray [n_users] — dominant category index per user
    """
    rng = np.random.default_rng(seed)

    # --- Build item catalog (8 categories x 25 items = 200) ---
    items = []
    for cat_idx, cat in enumerate(CATEGORIES):
        for name in ITEM_NAMES[cat]:
            cat_onehot = np.zeros(N_CATS, dtype=np.float32)
            cat_onehot[cat_idx] = 1.0

            price_tier = rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
            price_onehot = np.zeros(3, dtype=np.float32)
            price_onehot[price_tier] = 1.0

            popularity = float(rng.beta(2, 5))   # most items low popularity
            avg_rating = float(rng.uniform(0.5, 1.0))

            items.append({
                'name': name,
                'category': cat,
                'cat_idx': cat_idx,
                'price_tier': price_tier,
                # features layout: [cat_onehot(8), price_onehot(3), popularity(1), avg_rating(1)] = 13
                'features': np.concatenate([cat_onehot, price_onehot, [popularity, avg_rating]]).astype(np.float32),
            })
    items = items[:n_items]

    # --- Assign user archetypes (each user has a dominant category) ---
    user_archetypes = rng.integers(0, N_CATS, size=n_users)

    # --- Generate interactions biased toward each user's dominant category ---
    user_purchase_counts = np.zeros((n_users, N_CATS), dtype=np.float32)
    interactions = []

    # Pre-group item indices by category for fast sampling
    cat_item_indices = {
        cat_idx: [i for i, it in enumerate(items) if it['cat_idx'] == cat_idx]
        for cat_idx in range(N_CATS)
    }

    for _ in range(n_interactions):
        user_id = int(rng.integers(0, n_users))
        dominant_cat = user_archetypes[user_id]

        # Bias toward dominant category (~63% of purchases)
        cat_probs = np.ones(N_CATS) * 0.05
        cat_probs[dominant_cat] = 0.6
        cat_probs /= cat_probs.sum()
        chosen_cat = int(rng.choice(N_CATS, p=cat_probs))

        cat_items = cat_item_indices[chosen_cat]
        if not cat_items:
            continue
        item_id = int(rng.choice(cat_items))
        user_purchase_counts[user_id, chosen_cat] += 1
        interactions.append((user_id, item_id))

    # --- Build user feature vectors [n_users, 20] ---
    user_features = []
    for u in range(n_users):
        counts = user_purchase_counts[u]
        total = float(counts.sum())

        # log1p-normalized purchase counts per category
        log_counts = np.log1p(counts)
        norm_counts = log_counts / (log_counts.max() + 1e-8)

        # fraction of purchases per category (sums to 1)
        prefs = counts / (total + 1e-8)

        # scalar signals
        total_inter = np.log1p(total) / np.log1p(n_interactions / n_users * 2)
        total_inter = float(np.clip(total_inter, 0, 1))
        recency = float(rng.uniform(0, 1))       # synthetic
        price_sens = float(rng.uniform(0, 1))    # synthetic

        # Shannon entropy of purchase distribution (normalized to [0,1])
        p = prefs + 1e-10
        entropy = float(-np.sum(p * np.log(p)))
        variety = entropy / np.log(N_CATS)

        # layout: [norm_counts(8), prefs(8), total_inter(1), recency(1), price_sens(1), variety(1)] = 20
        feat = np.concatenate([norm_counts, prefs, [total_inter, recency, price_sens, variety]]).astype(np.float32)
        user_features.append(feat)

    user_features = np.array(user_features)
    item_features = np.array([it['features'] for it in items])

    return user_features, item_features, items, interactions, user_archetypes


class InteractionDataset(Dataset):
    """Returns (user_feature_vector, item_feature_vector) pairs."""

    def __init__(self, user_features, item_features, interactions):
        self.user_features = torch.tensor(user_features)
        self.item_features = torch.tensor(item_features)
        self.interactions = interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user_id, item_id = self.interactions[idx]
        return self.user_features[user_id], self.item_features[item_id]
