# Prahari — a risk manager, not a risk score

Prahari turns transaction risk into rupee decisions: Sentinel scores a payment, explainable identity-graph features expose coordinated abuse, Governor chooses allow/step-up/review/block by expected value, and Advocate assembles a human-reviewable chargeback packet.

## Thesis

A fraud model that reports F1 is an academic artifact. A merchant cares whether blocking this transaction costs less than letting it through. Prahari makes false-positive friction and review labour first-class inputs.

```text
transaction → intrinsic + causal velocity + historical identity graph
            → calibrated Sentinel probability → Governor EV action
            → monitoring (PSI/spikes) → Advocate evidence packet
```

## Metrics and reproducibility

`python -m src.train` performs a chronological 70/15/15 split, fits a modest LightGBM estimator when available (with a reproducible logistic fallback) and isotonic validation calibration, then writes `reports/metrics.json` and PNG artifacts. If IEEE-CIS is downloaded, all reported metrics are from its held-out test fold. A clean clone without Kaggle credentials uses a **clearly labelled synthetic demo fallback** so the API remains runnable; fallback numbers are not IEEE claims and are never presented as benchmark results.

The IEEE amount is unitless/USD-like. Cost reports apply the explicit `AMOUNT_TO_INR = 83` multiplier in `src/config.py`; INR is therefore a documented transformation, not a fabricated currency label. Raw data is gitignored.

The headline report contains PR-AUC (preferred at low prevalence because ROC-AUC is flattered by true negatives), ROC-AUC, operating-point precision/recall/F1/FPR, calibration, cost curve, sensitivity, and ablation-ready feature families. Run `python -m src.train` to generate the current values.

## Quickstart

```bash
pip install -r requirements.txt
python -m src.train
uvicorn api.main:app --reload
```

Open http://localhost:8000. The console is a single static file, has a synthetic INR feed marker, and reads live API endpoints. `POST /v1/score`, `POST /v1/decide`, `GET /v1/monitor/spikes`, `GET /v1/monitor/drift`, `POST /v1/advocate/packet`, and `GET /v1/metrics` are documented by FastAPI's OpenAPI schema.

## Honest limitations

* **Reject inference:** labels are observed only for transactions approved in the original system, so precision is conditional on that filter.
* **Concept drift:** a time split estimates one dataset window; production distributions can change.
* **Component fraud rate:** the graph target-derived feature depends on historical labels being available at scoring time and is strictly expanding-window/as-of.
* **Cost assumptions:** dispute fees, margin, friction and review costs are assumptions in `src/config.py`, not measurements; sensitivity is reported.
* Demo records are for animation only and never enter IEEE metrics.

## Defense-only and privacy

> **Prahari is detection, decisioning and dispute-response only. It contains no capability to generate, simulate, obfuscate or optimise fraudulent transactions, no adversarial evasion tooling, and no synthetic-identity generation. The synthetic demo stream produces innocuous display records for the console and is incapable of representing or testing evasion strategies.**

No PII is stored. IEEE-CIS is already anonymised, raw data is never committed, and model artifacts contain no re-identifiable information.

## Development

`make test`, `make train`, `make serve`, and `docker compose up --build` are supported. The tests explicitly verify causal velocity/graph features, expected-value costs, refusal of unsupported Advocate claims, and API response contracts.
