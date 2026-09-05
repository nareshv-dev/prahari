from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.config import MODEL_VERSION
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
    score = float(np.clip(.08 + .00003*amount + .12*bool(f.get("ring_score", 0)) + .08*bool(f.get("velocity_count_1h", 0) > 3), 0, 0.99))
    segment = req.merchant_category
    gov = governor_decide(score, amount, segment, thresholds())
    reasons = [{"feature": "amount", "contribution": round(.00003*amount, 5)},
               {"feature": "ring_score", "contribution": round(.12*float(f.get("ring_score", 0)), 5)},
               {"feature": "velocity_count_1h", "contribution": round(.08*float(f.get("velocity_count_1h", 0) > 3), 5)}]
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
    return {"synthetic": True, "categories": spike_report(d)}

def pd_date_range(n):
    import pandas as pd
    return pd.date_range("2026-01-01", periods=n, freq="h")

@app.get("/v1/monitor/drift")
def drift():
    return {"source": "synthetic-demo", "features": {}}

@app.get("/v1/metrics")
def metrics():
    p = Path("reports/metrics.json")
    return json.loads(p.read_text()) if p.exists() else {"status": "not-trained", "message": "Run python -m src.train"}

@app.post("/v1/advocate/packet")
def advocate(req: AdvocateRequest):
    return build_packet(_store, req.transaction_id)

web_dir = Path("web")
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
