"""Dependency helpers kept small so API tests can inject a model."""
from pathlib import Path
import json
from src.demo_data import generate_demo
from src.features.store import FeatureStore

def demo_store():
    d = generate_demo(300)
    d["DeviceInfo"] = "Chrome Windows"; d["card1"] = range(len(d)); d["addr1"] = [i % 50 for i in range(len(d))]
    return FeatureStore(d.rename(columns={"amount_inr":"TransactionAmt"}))

def thresholds():
    p = Path("models/thresholds.json")
    return json.loads(p.read_text()) if p.exists() else {"default": .5}
