"""Strictly backwards-looking velocity features."""
from __future__ import annotations
import numpy as np
import pandas as pd

KEYS = {"card1": "card1", "addr1": "addr1", "email": "P_emaildomain"}

def build_velocity(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "TransactionDT" in x:
        raw_order = pd.to_numeric(x["TransactionDT"], errors="coerce").to_numpy()
        order = np.where(np.isnan(raw_order), np.arange(len(x)), raw_order)
        # Preserve caller's chronological ordering contract while returning row order.
        sort_idx = np.argsort(order, kind="mergesort")
    else:
        sort_idx = np.arange(len(x))
    s = x.iloc[sort_idx].reset_index(drop=True)
    times = pd.to_numeric(s.get("TransactionDT", pd.Series(np.arange(len(s))*60)), errors="coerce").fillna(0)
    amount = pd.to_numeric(s.get("TransactionAmt", pd.Series(0, index=s.index)), errors="coerce").fillna(0)
    device = s.get("DeviceInfo", pd.Series("__missing__", index=s.index)).fillna("__missing__").astype(str)
    result = pd.DataFrame(0.0, index=s.index, columns=[
        f"{k}_{metric}_{w}" for k in KEYS for metric in ("count","amount_sum","amount_mean","distinct_device","distinct_amount","since_previous") for w in ("1h","24h","7d")
    ])
    for key_name, col in KEYS.items():
        vals = s.get(col, pd.Series("__missing__", index=s.index)).fillna("__missing__").astype(str)
        history = {}
        for i, (key, t, a, dev) in enumerate(zip(vals, times, amount, device)):
            rows = history.get(key, [])
            for label, window in (("1h", 3600), ("24h", 86400), ("7d", 604800)):
                prior = [(pt, pa, pdv) for pt, pa, pdv in rows if pt < t and t - pt <= window]
                result.loc[i, f"{key_name}_count_{label}"] = len(prior)
                result.loc[i, f"{key_name}_amount_sum_{label}"] = sum(v[1] for v in prior)
                result.loc[i, f"{key_name}_amount_mean_{label}"] = np.mean([v[1] for v in prior]) if prior else 0
                result.loc[i, f"{key_name}_distinct_device_{label}"] = len({v[2] for v in prior})
                result.loc[i, f"{key_name}_distinct_amount_{label}"] = len({v[1] for v in prior})
                result.loc[i, f"{key_name}_since_previous_{label}"] = (t - rows[-1][0]) if rows else 0
            history.setdefault(key, []).append((t, a, dev))
    result = result.iloc[np.argsort(sort_idx)].reset_index(drop=True)
    return result
