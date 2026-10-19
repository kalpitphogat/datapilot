"""Generates sample_data/sales.csv: synthetic Indian e-commerce sales with real patterns to discover.

Planted patterns (for demos):
- Festive spike in October-November (Diwali season).
- East region revenue declines steadily through 2025.
- Marketing spend correlates with revenue, with diminishing returns.
- Electronics has noticeably lower customer ratings than other categories.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

REGIONS = ["North", "South", "East", "West"]
CATEGORIES = {
    "Electronics": (["Earbuds", "Smartwatch", "Power Bank", "Bluetooth Speaker"], 2500),
    "Fashion": (["Kurta", "Sneakers", "Saree", "Denim Jacket"], 1200),
    "Home & Kitchen": (["Mixer Grinder", "Pressure Cooker", "Bedsheet Set", "Water Purifier"], 1800),
    "Books": (["Exam Guide", "Novel", "Self-Help", "Comics"], 350),
    "Beauty": (["Face Serum", "Sunscreen", "Hair Oil", "Lipstick"], 600),
}

dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
rows = []
for date in dates:
    festive = 1.8 if date.month in (10, 11) else 1.0
    for _ in range(rng.poisson(6)):
        region = rng.choice(REGIONS)
        category = rng.choice(list(CATEGORIES))
        products, base_price = CATEGORIES[category]
        product = rng.choice(products)

        decline = 1 - 0.55 * (date.dayofyear / 365) if region == "East" else 1.0
        marketing = round(float(rng.gamma(2.0, 1500)), 2)
        demand = festive * decline * (1 + 0.6 * np.log1p(marketing / 1000))
        units = max(1, int(rng.poisson(3 * demand)))
        price = round(base_price * rng.uniform(0.8, 1.2), 2)

        mean_rating = 3.4 if category == "Electronics" else 4.2
        rating = float(np.clip(rng.normal(mean_rating, 0.5), 1, 5).round(1))

        rows.append({
            "date": date.date(),
            "region": region,
            "category": category,
            "product": product,
            "units_sold": units,
            "unit_price": price,
            "revenue": round(units * price, 2),
            "marketing_spend": marketing,
            "customer_rating": rating,
        })

out = Path(__file__).resolve().parent.parent / "sample_data" / "sales.csv"
out.parent.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Wrote {len(rows)} rows to {out}")
