"""Reproducible chronological training pipeline with an IEEE-or-demo fallback."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.isotonic import IsotonicRegression
from .config import MODEL_DIR, REPORT_DIR, SEED
from .demo_data import generate_demo
from .features.intrinsic import build_intrinsic
from .features.velocity import build_velocity
from .features.graph import build_graph

def load_dataset(path="data/raw"):
    p = Path(path)
    train = p / "train_transaction.csv"
    if train.exists():
        return pd.read_csv(train), "IEEE-CIS"
    # Clearly marked fallback for clean clones without Kaggle credentials.
    d = generate_demo(1500, SEED).rename(columns={"amount_inr": "TransactionAmt"})
    d["TransactionDT"] = np.arange(len(d)) * 3600
    d["card1"] = np.arange(len(d)) % 300; d["addr1"] = np.arange(len(d)) % 120
    d["P_emaildomain"] = np.where(np.arange(len(d)) % 4 == 0, "example.test", "mail.test")
    d["DeviceInfo"] = np.where(np.arange(len(d)) % 5 == 0, "Chrome Windows", "Safari iOS")
    d["ProductCD"] = d["merchant_category"]
    return d, "synthetic-demo-fallback"

def chronological_split(df):
    x = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    n = len(x); a, b = int(n*.70), int(n*.85)
    return x.iloc[:a], x.iloc[a:b], x.iloc[b:]

def make_features(df, family="all"):
    parts = [build_intrinsic(df)]
    if family in ("velocity", "graph", "all"): parts.append(build_velocity(df))
    if family in ("graph", "all"): parts.append(build_graph(df))
    out = pd.concat(parts, axis=1)
    # sklearn-friendly deterministic encoding; categorical columns are retained as strings.
    return out.loc[:, ~out.columns.duplicated()]

def _pipeline(X):
    cats = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    nums = [c for c in X.columns if c not in cats]
    prep = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                                 ("scale", StandardScaler())]), nums),
                             ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                                ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats)])
    try:
        from lightgbm import LGBMClassifier
        estimator = LGBMClassifier(n_estimators=180, learning_rate=.05, num_leaves=24,
                                   subsample=.85, colsample_bytree=.85, random_state=SEED,
                                   verbosity=-1, class_weight="balanced")
    except ImportError:
        estimator = LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED)
    return Pipeline([("prep", prep), ("model", estimator)])

def train(output_dir=MODEL_DIR):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True); REPORT_DIR.mkdir(exist_ok=True)
    df, source = load_dataset()
    train_df, val_df, test_df = chronological_split(df)
    y = df["isFraud"].astype(int).to_numpy()
    X_all = make_features(df)
    Xtr, Xv, Xte = X_all.iloc[:len(train_df)], X_all.iloc[len(train_df):len(train_df)+len(val_df)], X_all.iloc[len(train_df)+len(val_df):]
    model = _pipeline(Xtr); model.fit(Xtr, train_df["isFraud"].astype(int))
    raw_val = model.predict_proba(Xv)[:, 1]; raw_test = model.predict_proba(Xte)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_val, val_df["isFraud"].astype(int))
    calibrated = calibrator.predict(raw_test)
    joblib.dump(model, output_dir / "model.pkl"); joblib.dump(calibrator, output_dir / "calibrator.pkl")
    (output_dir / "lgbm.txt").write_text(
        "Prahari portable fallback estimator\n"
        "The clean-clone path uses sklearn logistic regression when LightGBM is unavailable.\n"
        "Kaggle runs may replace this artifact with the optional LightGBM model.\n")
    manifest = {"source": source, "features": list(X_all.columns), "model_version": "demo-v1",
                "train_rows": len(train_df), "validation_rows": len(val_df), "test_rows": len(test_df)}
    (output_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))
    from .evaluate import evaluate
    from .config import AMOUNT_TO_INR
    amounts = test_df.get("TransactionAmt", pd.Series(1, index=test_df.index)).to_numpy()
    if source == "IEEE-CIS":
        amounts = amounts * AMOUNT_TO_INR
    metrics = evaluate(test_df["isFraud"].to_numpy(), calibrated, amounts, source=source)
    from .governor import save_thresholds
    save_thresholds({"default": metrics["cost_optimum"]["threshold"]})
    # Mechanical ablation proof: use the same estimator and chronological fold.
    ablation = {}
    for family in ("intrinsic", "velocity", "graph"):
        xa = make_features(df, family)
        ma = _pipeline(xa.iloc[:len(train_df)]); ma.fit(xa.iloc[:len(train_df)], train_df["isFraud"].astype(int))
        pa = ma.predict_proba(xa.iloc[len(train_df)+len(val_df):])[:, 1]
        neg = pa[y[len(train_df)+len(val_df):] == 0]
        cutoff = np.quantile(neg, .99) if len(neg) else 1
        yy = y[len(train_df)+len(val_df):]
        ablation[family] = {"recall_at_1pct_fpr": float(((pa >= cutoff) & (yy == 1)).sum()/max(1,(yy == 1).sum()))}
    metrics["ablation_recall_at_1pct_fpr"] = ablation
    (REPORT_DIR / "ablation.json").write_text(json.dumps(ablation, indent=2))
    # PSI is computed on intrinsic numeric columns between train and held-out windows.
    from .monitor import drift_report
    numeric_train = make_features(train_df, "intrinsic").select_dtypes("number")
    numeric_test = make_features(test_df, "intrinsic").select_dtypes("number")
    drift = drift_report(numeric_train, numeric_test)
    (REPORT_DIR / "drift_report.json").write_text(json.dumps(drift, indent=2))
    try:
        import matplotlib.pyplot as plt
        names, values = list(drift), [v["psi"] for v in drift.values()]
        plt.figure(figsize=(7, 3)); plt.bar(names, values, color="#E8A33D"); plt.axhline(.2, color="#E2604A", linestyle="--")
        plt.xticks(rotation=70, ha="right", fontsize=7); plt.ylabel("PSI"); plt.tight_layout(); plt.savefig(REPORT_DIR/"drift_psi.png", dpi=120); plt.close()
    except Exception:
        pass
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
