"""Operational drift and category spike monitoring."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import PSI_ALERT, SPIKE_Z_ALERT

def psi(expected, actual, bins=10):
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if not len(e) or not len(a): return 0.0
    cuts = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 2: return 0.0
    ep, _ = np.histogram(e, cuts); ap, _ = np.histogram(a, cuts)
    ep = (ep + 1) / (ep.sum() + len(ep)); ap = (ap + 1) / (ap.sum() + len(ap))
    return float(np.sum((ap - ep) * np.log(ap / ep)))

def drift_report(training: pd.DataFrame, current: pd.DataFrame, features=None):
    features = features or [c for c in training.columns if c in current.columns and pd.api.types.is_numeric_dtype(training[c])]
    return {c: {"psi": psi(training[c].dropna(), current[c].dropna()), "alert": psi(training[c].dropna(), current[c].dropna()) >= PSI_ALERT} for c in features}

def spike_report(events: pd.DataFrame, category_col="merchant_category", time_col="timestamp", flagged_col="flagged", window="1h"):
    if events.empty: return {}
    x = events.copy(); x[time_col] = pd.to_datetime(x[time_col]); x["window"] = x[time_col].dt.floor(window)
    grouped = x.groupby(["window", category_col])[flagged_col].mean().reset_index()
    out = {}
    for cat, g in grouped.groupby(category_col):
        values = g[flagged_col].to_numpy(float); mean = values[:-1].mean() if len(values)>1 else values.mean()
        std = values[:-1].std() if len(values)>2 else 1e-9
        z = float((values[-1]-mean)/std) if std else 0.0
        out[str(cat)] = {"z_score": z, "alert": z >= SPIKE_Z_ALERT, "flag_rate": float(values[-1])}
    return out
