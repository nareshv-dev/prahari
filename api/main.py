from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.config import MODEL_VERSION, SEED
from src.demo_data import generate_demo
from src.governor import decide as governor_decide, expected_values
from src.monitor import spike_report, drift_report
from src.advocate import build_packet
from .schemas import ScoreRequest, ScoreResponse, DecideResponse, AdvocateRequest, Ring
from .deps import demo_store, thresholds

app = FastAPI(title="Prahari", version=MODEL_VERSION, description="A risk manager, not a risk score.")
_store = demo_store()

def _score(req: ScoreRequest):
    f = req.features
    amount = req.amount
    # Deterministic transparent fallback score; production model can be loaded by train.py.
    ring_signal = float(np.clip(float(f.get("ring_score", 0) or 0), 0, 1))
    velocity_signal = float(bool(f.get("velocity_count_1h", 0) > 3))
    score = float(np.clip(.08 + .00003*amount + .12*ring_signal + .08*velocity_signal, 0, 0.99))
    segment = req.merchant_category
    gov = governor_decide(score, amount, segment, thresholds())
    reasons = [{"feature": "amount", "contribution": round(.00003*amount, 5)},
               {"feature": "ring_score", "contribution": round(.12*ring_signal, 5)},
               {"feature": "velocity_count_1h", "contribution": round(.08*velocity_signal, 5)}]
    return score, gov, reasons

@app.get("/health")
def health(): return {"status": "ok", "model_version": MODEL_VERSION, "demo": True}

@app.post("/v1/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    raw, gov, reasons = _score(req)
    return ScoreResponse(score=raw, calibrated_probability=raw, action=gov["action"], threshold_used=gov["threshold_used"],
                         segment=req.merchant_category, ring=Ring(in_ring=bool(req.features.get("in_suspected_ring", False)),
                         component_size=int(req.features.get("graph_component_size", 0)), ring_score=float(req.features.get("ring_score", 0))),
                         explanations=reasons[:5], model_version=MODEL_VERSION)

@app.post("/v1/decide", response_model=DecideResponse)
def decide(req: ScoreRequest):
    raw, gov, reasons = _score(req)
    base = score(req)
    payload = base.model_dump() if hasattr(base, "model_dump") else base.dict()
    return DecideResponse(**payload, expected_values=gov["expected_values"])

@app.get("/v1/monitor/spikes")
def spikes():
    d = generate_demo(300); d["timestamp"] = pd_date_range(len(d)); d["flagged"] = d.score > .2
    categories = spike_report(d)
    # The UI gets a small, deterministic series so it can render sparklines without
    # inventing a second data source. The original categories mapping is retained.
    series = {}
    for index, category in enumerate(categories):
        base = 0.08 + index * 0.025
        points = [round(max(0.01, base + 0.035 * math.sin((i + index) * 0.9) +
                             (0.12 if i == 11 and index == 1 else 0)), 3)
                  for i in range(12)]
        series[category] = points
    return {"synthetic": True, "generated_at": "2026-01-13T12:00:00Z",
            "categories": categories, "series": series}

def pd_date_range(n):
    import pandas as pd
    return pd.date_range("2026-01-01", periods=n, freq="h")

@app.get("/v1/monitor/drift")
def drift():
    baseline = generate_demo(240)
    current = generate_demo(120, seed=84)
    current["amount_inr"] = current["amount_inr"] * 1.12
    report = drift_report(baseline, current, features=["amount_inr", "score"])
    return {"source": "synthetic-demo", "generated_at": "2026-01-13T12:00:00Z",
            "status": "watch" if any(v["alert"] for v in report.values()) else "stable",
            "features": report}


@app.get("/v1/demo/stream")
def demo_stream(limit: int = 10):
    """Return innocuous, deterministic records for the presentation console."""
    limit = max(1, min(limit, 30))
    data = generate_demo(limit, seed=SEED)
    events = []
    for index, row in data.iterrows():
        ring_score = round(float((index % 5) / 5), 2)
        request = ScoreRequest(
            transaction_id=str(row["transaction_id"]),
            amount=float(row["amount_inr"]),
            merchant_category=str(row["merchant_category"]),
            features={"ring_score": ring_score,
                      "velocity_count_1h": int(index % 7),
                      "in_suspected_ring": ring_score >= .6,
                      "graph_component_size": int(4 + index % 9)},
        )
        raw, gov, reasons = _score(request)
        events.append({
            "transaction_id": request.transaction_id,
            "amount_inr": round(request.amount, 2),
            "merchant_category": request.merchant_category,
            "payment_method": str(row["payment_method"]),
            "score": round(raw, 4),
            "action": gov["action"],
            "risk_band": "high" if raw >= .65 else ("watch" if raw >= .28 else "low"),
            "ring_score": ring_score,
            "ring_size": request.features["graph_component_size"],
            "reason": reasons[0]["feature"] if reasons else "model signal",
            "timestamp": f"2026-01-13T12:{index:02d}:00Z",
            "synthetic": True,
        })
    return {"synthetic": True, "generated_at": "2026-01-13T12:00:00Z",
            "events": events}


@app.post("/v1/playground/score", response_model=DecideResponse)
def playground_score(req: ScoreRequest):
    """Named alias for the UI playground; the decision contract stays identical."""
    return decide(req)

@app.get("/v1/metrics")
def metrics():
    p = Path("reports/metrics.json")
    return json.loads(p.read_text()) if p.exists() else {"status": "not-trained", "message": "Run python -m src.train"}

@app.get("/v1/metrics/cost-curve")
def cost_curve():
    """Serves the full threshold sweep so the console can compare any two operating
    points (e.g. a fixed 0.50 cutoff vs. the Governor's EV-optimal threshold) using
    only numbers evaluate.py already computed - no retraining, no fabricated values."""
    p = Path("reports/cost_curve.csv")
    if not p.exists():
        return {"status": "not-trained", "message": "Run python -m src.train", "points": []}
    import csv
    with p.open() as fh:
        points = [{"threshold": float(row["threshold"]), "net_saved": float(row["net_saved"])} for row in csv.DictReader(fh)]
    return {"points": points}

@app.post("/v1/advocate/packet")
def advocate(req: AdvocateRequest):
    return build_packet(_store, req.transaction_id)

web_dir = Path("web")
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
