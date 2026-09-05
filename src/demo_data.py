"""Clearly synthetic INR stream for the console only (never used in reports)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import SEED

def generate_demo(n: int = 120, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = np.array(["fashion", "electronics", "food", "travel", "subscriptions"])
    methods = np.array(["UPI", "card", "netbanking"])
    amount = np.exp(rng.normal(np.log(900), 0.9, n)).round(2)
    fraud = rng.random(n) < (.035 + (amount > 4000) * .025)
    return pd.DataFrame({
        "transaction_id": [f"demo-{i:05d}" for i in range(n)],
        "amount_inr": amount, "merchant_category": rng.choice(categories, n),
        "payment_method": rng.choice(methods, n, p=[.55, .35, .10]),
        "score": np.clip(.12 * fraud + rng.beta(1.5, 10, n), 0, 1),
        "isFraud": fraud.astype(int), "synthetic": True,
    })

