"""Transaction-intrinsic features; no target or future observations are used."""
from __future__ import annotations
import numpy as np
import pandas as pd

def build_intrinsic(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    out = pd.DataFrame(index=x.index)
    numeric = ["TransactionAmt", "card1", "card2", "card3", "card4", "card5", "card6",
               "addr1", "addr2", "dist1", "dist2"]
    for c in numeric:
        out[c] = pd.to_numeric(x[c], errors="coerce") if c in x else np.nan
    out["amount"] = out["TransactionAmt"]
    out["log_amount"] = np.log1p(out["TransactionAmt"].clip(lower=0))
    if "TransactionDT" in x:
        dt = pd.to_numeric(x["TransactionDT"], errors="coerce")
        out["hour"] = (dt // 3600) % 24
        out["day_of_week"] = (dt // 86400) % 7
    else:
        out[["hour", "day_of_week"]] = 0
    for c in ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
              "DeviceType", "DeviceInfo"]:
        if c in x:
            out[c] = x[c].astype("string").fillna("__MISSING__")
    if "P_emaildomain" in x and "R_emaildomain" in x:
        out["email_match"] = (x["P_emaildomain"].fillna("") == x["R_emaildomain"].fillna("")).astype(int)
    else:
        out["email_match"] = 0
    if "DeviceInfo" in x:
        s = x["DeviceInfo"].fillna("").astype(str)
        out["browser"] = s.str.extract(r"(chrome|safari|firefox|edge|ie|opera)", flags=2, expand=False).fillna("unknown")
        out["os"] = s.str.extract(r"(windows|mac|ios|android|linux)", flags=2, expand=False).fillna("unknown")
    for c in ["V1", "V2", "V3", "C1", "C2", "D1", "D2", "M1", "M2", "M4", "M6", "M7", "M8", "M9"]:
        if c in x:
            out[c] = x[c]
    # Missingness is deliberately explicit: null blocks are informative in IEEE.
    for c in list(x.columns):
        if c in numeric or c in ("P_emaildomain", "R_emaildomain", "DeviceInfo", "DeviceType"):
            out[f"missing_{c}"] = x[c].isna().astype("int8")
    return out.reset_index(drop=True)

