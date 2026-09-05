"""Human-reviewable chargeback evidence packet and cited representment draft."""
from __future__ import annotations
from .features.store import FeatureStore

CLAIMS = [
    ("device_fingerprint", "The device fingerprint was observed in the transaction record ({device_fingerprint})."),
    ("identity_history", "The identity cluster has prior settled activity ({identity_history})."),
    ("address_bin_consistency", "Address and BIN fields were consistent with the recorded purchase ({address_bin_consistency})."),
    ("velocity_profile", "The purchase had the following historical velocity profile ({velocity_profile})."),
    ("sentinel_score", "Prahari recorded a calibrated risk score of {sentinel_score}."),
]

def evidence_packet(store: FeatureStore, transaction_id: str) -> dict:
    r = store.as_of(transaction_id); history = store.history(transaction_id)
    return {"transaction_id": transaction_id,
            "device_fingerprint": r.get("DeviceInfo"),
            "identity_history": f"{len(history)} transaction(s) available as-of purchase",
            "address_bin_consistency": f"addr1={r.get('addr1')}, card1={r.get('card1')}",
            "velocity_profile": r.get("velocity_summary"),
            "sentinel_score": r.get("calibrated_probability")}

def representment_draft(packet: dict) -> dict:
    sentences, citations = [], []
    for field, template in CLAIMS:
        value = packet.get(field)
        if value is None or value == "" or value == "None":
            continue
        sentences.append(template.format(**{field: value})); citations.append(field)
    return {"draft": " ".join(sentences), "citations": citations, "refused_fields": [f for f, _ in CLAIMS if f not in citations]}

def build_packet(store: FeatureStore, transaction_id: str):
    packet = evidence_packet(store, transaction_id)
    draft = representment_draft(packet)
    return {"evidence": packet, **draft}
