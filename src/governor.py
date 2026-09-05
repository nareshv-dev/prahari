"""Governor: calibrated probabilities to segment-aware rupee decisions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from . import costs
from .config import MODEL_VERSION, MODEL_DIR

ACTIONS = ("ALLOW", "STEP_UP", "REVIEW", "BLOCK")

def expected_values(probability: float, amount: float, friction_rate=None):
    kw = {} if friction_rate is None else {"friction_rate": friction_rate}
    return {a: float(costs.action_expected_cost(probability, amount, a, **kw)) for a in ACTIONS}

def decide(probability: float, amount: float, segment: str = "default", thresholds: dict | None = None):
    thresholds = thresholds or {}
    t = float(thresholds.get(segment, thresholds.get("default", .5)))
    # Minimum-cost action is preferable to a threshold-only rule.
    ev = expected_values(probability, amount)
    action = min(ev, key=ev.get)
    if probability < t and action == "BLOCK":
        action = "REVIEW" if probability >= t*.65 else "ALLOW"
    return {"action": action, "threshold_used": t, "expected_values": ev, "segment": segment,
            "model_version": MODEL_VERSION}

def fit_segment_thresholds(y_true, probability, amount, segments, grid=None):
    grid = np.linspace(.01, .99, 99) if grid is None else np.asarray(grid)
    out = {}
    for segment in sorted(set(map(str, segments))):
        mask = np.asarray(segments).astype(str) == segment
        if mask.any():
            out[segment] = costs.optimum(np.asarray(y_true)[mask], np.asarray(probability)[mask],
                                         np.asarray(amount)[mask], thresholds=grid)[0]
    return out

def save_thresholds(thresholds, path: Path = MODEL_DIR / "thresholds.json"):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(thresholds, indent=2))

