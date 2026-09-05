"""Strictly backwards-looking velocity features.

Implemented with per-key time-ordered lists and binary search (bisect) instead of
re-scanning full key history per row: each row only touches the slice of its key's
history that falls inside the current window, with O(1) prefix-sum lookups for the
sum/count metrics. This keeps the pass roughly O(n log n) instead of O(n * history),
which matters once n is in the hundreds of thousands (IEEE-CIS is ~590k rows).
"""
from __future__ import annotations
import bisect
import numpy as np
import pandas as pd

KEYS = {"card1": "card1", "addr1": "addr1", "email": "P_emaildomain"}
WINDOWS = (("1h", 3600.0), ("24h", 86400.0), ("7d", 604800.0))

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
    n = len(s)
    times = pd.to_numeric(s.get("TransactionDT", pd.Series(np.arange(n) * 60)), errors="coerce").fillna(0).to_numpy(dtype=float)
    amount = pd.to_numeric(s.get("TransactionAmt", pd.Series(0, index=s.index)), errors="coerce").fillna(0).to_numpy(dtype=float)
    device = s.get("DeviceInfo", pd.Series("__missing__", index=s.index)).fillna("__missing__").astype(str).to_numpy()

    columns = {f"{k}_{metric}_{label}": np.zeros(n, dtype=float)
               for k in KEYS for metric in ("count", "amount_sum", "amount_mean", "distinct_device", "distinct_amount", "since_previous")
               for label, _ in WINDOWS}

    for key_name, col in KEYS.items():
        vals = s.get(col, pd.Series("__missing__", index=s.index)).fillna("__missing__").astype(str).to_numpy()
        key_times: dict[str, list] = {}
        key_amount: dict[str, list] = {}
        key_device: dict[str, list] = {}
        key_cum_amount: dict[str, list] = {}
        count_cols = {label: columns[f"{key_name}_count_{label}"] for label, _ in WINDOWS}
        sum_cols = {label: columns[f"{key_name}_amount_sum_{label}"] for label, _ in WINDOWS}
        mean_cols = {label: columns[f"{key_name}_amount_mean_{label}"] for label, _ in WINDOWS}
        ddev_cols = {label: columns[f"{key_name}_distinct_device_{label}"] for label, _ in WINDOWS}
        damt_cols = {label: columns[f"{key_name}_distinct_amount_{label}"] for label, _ in WINDOWS}
        since_cols = {label: columns[f"{key_name}_since_previous_{label}"] for label, _ in WINDOWS}
        for i in range(n):
            key = vals[i]
            t = times[i]
            kt = key_times.get(key)
            if kt is None:
                kt = []; key_times[key] = kt
                key_amount[key] = []
                key_device[key] = []
                key_cum_amount[key] = [0.0]
            ka, kd, kc = key_amount[key], key_device[key], key_cum_amount[key]
            since = (t - kt[-1]) if kt else 0.0
            for label, window in WINDOWS:
                hi = bisect.bisect_left(kt, t)
                lo = bisect.bisect_left(kt, t - window)
                cnt = hi - lo
                asum = kc[hi] - kc[lo]
                count_cols[label][i] = cnt
                sum_cols[label][i] = asum
                mean_cols[label][i] = (asum / cnt) if cnt else 0.0
                ddev_cols[label][i] = len(set(kd[lo:hi]))
                damt_cols[label][i] = len(set(ka[lo:hi]))
                since_cols[label][i] = since
            kt.append(t)
            ka.append(amount[i])
            kd.append(device[i])
            kc.append(kc[-1] + amount[i])

    result = pd.DataFrame(columns)
    result = result.iloc[np.argsort(sort_idx)].reset_index(drop=True)
    return result
