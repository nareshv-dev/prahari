"""Held-out metrics and report artifact generation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix, brier_score_loss
from .config import REPORT_DIR, OPERATING_POINTS
from .costs import optimum, sensitivity

def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins+1); total = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.any(): total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)

def evaluate(y_true, probability, amount, source="IEEE-CIS", output_dir=REPORT_DIR):
    output_dir = Path(output_dir); output_dir.mkdir(exist_ok=True)
    y, p, a = np.asarray(y_true).astype(int), np.asarray(probability), np.asarray(amount)
    metrics = {"source": source, "n_test": int(len(y)), "prevalence": float(y.mean()),
               "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5,
               "brier_score": float(brier_score_loss(y, p)), "ece": _ece(y, p), "operating_points": {}}
    for name, t in OPERATING_POINTS.items():
        pred = p >= t
        pr, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        metrics["operating_points"][name] = {"threshold": t, "precision": float(pr), "recall": float(rec), "f1": float(f1), "fpr": float(fp/max(1, tn+fp))}
    t, saved, curve = optimum(y, p, a)
    metrics["cost_optimum"] = {"threshold": t, "net_saved": saved, "saved_per_10000": saved/len(y)*10000}
    metrics["sensitivity"] = {k: {"threshold": v[0], "saved_per_10000": v[1]/len(y)*10000} for k,v in sensitivity(y,p,a).items()}
    metrics["limitations"] = [
        "Precision is conditional on labels observed after the original approval filter (reject inference).",
        "Concept drift can invalidate a historical time window.",
        "Cost constants are assumptions, not measurements."
    ]
    (output_dir / "cost_curve.csv").write_text(curve.to_csv(index=False))
    try:
        import matplotlib.pyplot as plt
        plt.figure(); plt.plot(curve.threshold, curve.net_saved); plt.axvline(t, color="#E2604A"); plt.xlabel("Threshold"); plt.ylabel("Net INR saved"); plt.tight_layout(); plt.savefig(output_dir/"cost_curve.png", dpi=120); plt.close()
        plt.figure(); plt.hist(p[y==0], bins=20, alpha=.6, label="legitimate"); plt.hist(p[y==1], bins=20, alpha=.6, label="fraud"); plt.legend(); plt.tight_layout(); plt.savefig(output_dir/"score_distribution.png", dpi=120); plt.close()
        # Reliability diagram is intentionally simple and machine-independent.
        edges = np.linspace(0, 1, 11); centers=[]; observed=[]
        for lo, hi in zip(edges[:-1], edges[1:]):
            m=(p>=lo)&(p<hi)
            if m.any(): centers.append(float(p[m].mean())); observed.append(float(y[m].mean()))
        plt.figure(); plt.plot([0,1],[0,1],"--",color="#7E9094"); plt.plot(centers,observed,"o-",color="#4FB49A")
        plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency"); plt.tight_layout(); plt.savefig(output_dir/"calibration.png", dpi=120); plt.close()
        pd = __import__("pandas")
        pd.DataFrame({"bin_prediction": centers, "observed_frequency": observed}).to_csv(output_dir/"calibration.csv", index=False)
    except Exception:
        pass
    return metrics
