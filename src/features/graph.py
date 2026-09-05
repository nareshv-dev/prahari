"""Incremental identity graph features with strict historical semantics."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import numpy as np
import pandas as pd

def _union_find_features(x: pd.DataFrame) -> pd.DataFrame:
    parent, size = {}, {}
    labels = defaultdict(lambda: [0, 0])
    artifact_seen = defaultdict(set)
    artifact_degree = defaultdict(int)
    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if size.get(ra, 1) < size.get(rb, 1): ra, rb = rb, ra
            parent[rb] = ra; size[ra] = size.get(ra, 1) + size.get(rb, 1)
            labels[ra][0] += labels[rb][0]
            labels[ra][1] += labels[rb][1]
    rows = []
    for _, r in x.iterrows():
        aid = []
        for name in ("card1", "addr1", "P_emaildomain", "DeviceInfo"):
            val = r.get(name)
            if pd.notna(val) and str(val) != "":
                aid.append(f"{name}:{val}")
        aid.append(f"card_addr:{r.get('card1', '__missing__')}|{r.get('addr1', '__missing__')}")
        tid = f"txn:{len(rows)}"
        # Snapshot before adding current transaction: no current label leakage.
        roots = [find(a) for a in aid if a in parent]
        root = roots[0] if roots else tid
        prior_size = size.get(root, 1) if roots else 0
        hist_fraud = labels[root][0] / labels[root][1] if labels[root][1] else 0.0
        cards = set(); emails = set(); addrs = set()
        for a in aid:
            artifact_degree[a] += 0
            artifact_seen[a].add((r.get("card1"), r.get("P_emaildomain"), r.get("addr1")))
            if a.startswith("card1:"): cards.add(a)
            if a.startswith("P_emaildomain:"): emails.add(a)
            if a.startswith("addr1:"): addrs.add(a)
        density = min(1.0, len(roots) / max(1, len(aid)))
        ring_score = min(1.0, 0.5 * np.log1p(prior_size) / 5 + 0.5 * density)
        rows.append({"graph_component_size": prior_size, "graph_fraud_rate": hist_fraud,
                     "graph_density": density, "ring_score": ring_score,
                     "in_suspected_ring": bool(prior_size >= 3 and density >= .2),
                     "card_degree": artifact_degree.get(f"card1:{r.get('card1')}", 0),
                     "addr_degree": artifact_degree.get(f"addr1:{r.get('addr1')}", 0),
                     "email_degree": artifact_degree.get(f"P_emaildomain:{r.get('P_emaildomain')}", 0),
                     "device_degree": artifact_degree.get(f"DeviceInfo:{r.get('DeviceInfo')}", 0)})
        for a in aid:
            union(tid, a)
            artifact_degree[a] += 1
        root = find(tid)
        # Historical labels are only incorporated after the row's features are emitted.
        label = int(r.get("isFraud", 0) or 0)
        labels[find(root)][0] += label; labels[find(root)][1] += 1
    return pd.DataFrame(rows)

def build_graph(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "TransactionDT" in x:
        order = np.argsort(pd.to_numeric(x["TransactionDT"], errors="coerce").fillna(0).to_numpy(), kind="mergesort")
    else: order = np.arange(len(x))
    f = _union_find_features(x.iloc[order].reset_index(drop=True))
    inv = np.argsort(order)
    return f.iloc[inv].reset_index(drop=True)
