"""Expected-value costs and threshold sweeps in INR."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import C_FN_FIXED, C_CHARGEBACK_FEE, FP_MARGIN_RATE, FP_FRICTION_RATE, C_REVIEW, REVIEW_FRACTION

def false_positive_cost(amount, friction_rate=FP_FRICTION_RATE):
    return np.asarray(amount) * (FP_MARGIN_RATE + friction_rate)

def false_negative_cost(amount):
    return np.asarray(amount) + C_FN_FIXED + C_CHARGEBACK_FEE

def action_expected_cost(probability, amount, action, friction_rate=FP_FRICTION_RATE):
    p, a = np.asarray(probability), np.asarray(amount)
    if action == "ALLOW": return p * false_negative_cost(a)
    if action == "BLOCK": return (1-p) * false_positive_cost(a)
    if action == "REVIEW": return C_REVIEW + p * false_negative_cost(a) * (1-REVIEW_FRACTION)
    if action == "STEP_UP": return .5 * (p * false_negative_cost(a) + (1-p) * false_positive_cost(a))
    raise ValueError(f"Unknown action: {action}")

def savings_curve(y_true, probability, amount, thresholds=None, friction_rate=FP_FRICTION_RATE):
    thresholds = np.linspace(0, 1, 101) if thresholds is None else np.asarray(thresholds)
    y, p, a = map(np.asarray, (y_true, probability, amount))
    baseline = float(np.sum(y * false_negative_cost(a)))
    rows = []
    for t in thresholds:
        block = p >= t
        cost = np.sum((~block) * y * false_negative_cost(a) + block * (1-y) * false_positive_cost(a, friction_rate))
        rows.append({"threshold": float(t), "net_saved": float(baseline - cost)})
    return pd.DataFrame(rows)

def optimum(y_true, probability, amount, **kwargs):
    curve = savings_curve(y_true, probability, amount, **kwargs)
    row = curve.iloc[curve["net_saved"].argmax()]
    return float(row["threshold"]), float(row["net_saved"]), curve

def sensitivity(y_true, probability, amount):
    return {str(mult): optimum(y_true, probability, amount, friction_rate=FP_FRICTION_RATE*mult)[:2] for mult in (.5, 1, 2)}

