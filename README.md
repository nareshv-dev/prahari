# Prahari, a risk manager, not a risk score

Prahari turns transaction risk into rupee decisions. Sentinel scores a payment, explainable identity-graph features expose coordinated abuse rings, Governor chooses allow, step-up, review, or block by expected value, and Advocate assembles a human-reviewable chargeback packet when a dispute lands anyway.

Built for Razorpay's AI Risk Manager track: stop a merchant losing money to fraud, returns, and chargebacks.

## The problem

Most fraud tooling reports F1, precision, and recall, then stops. A merchant does not run on F1. A merchant runs on rupees: does blocking this transaction cost less than letting it through, given what a false positive actually costs in lost margin and customer friction. Almost every system treats that false-positive cost as a footnote. Prahari makes it a first-class input to the model's operating point instead.

## Thesis

A fraud model that reports F1 is an academic artifact. Prahari's Governor prices every decision in rupees, not accuracy, and picks the cutoff that maximises net rupees saved for that merchant segment rather than a generic 0.5 threshold.

```text
transaction -> intrinsic + causal velocity + historical identity graph
            -> calibrated Sentinel probability -> Governor EV action
            -> monitoring (PSI / spikes) -> Advocate evidence packet
```

Sentinel, Governor, and Advocate all read from one shared feature store with strict as-of-time semantics, so Advocate never sees information that was not yet available at the moment of purchase.

## Headline metrics

Trained and evaluated on the real IEEE-CIS Fraud Detection dataset (590,540 transactions, Kaggle/Vesta), split chronologically, never randomly, because a random split leaks future information through shared cards and devices and inflates every leaderboard number you have seen elsewhere.

| Fold | Day range (relative to first transaction) | Rows | Fraud rate |
|---|---|---|---|
| Train | day 1.0 to day 120.8 | 413,378 | 3.52% |
| Validation | day 120.8 to day 152.2 | 88,581 | 3.43% |
| Test (held out) | day 152.2 to day 183.0 | 88,581 | 3.48% |

| Metric | Value | Why it matters |
|---|---|---|
| PR-AUC | **0.369** | Leads the report. At 3.5% prevalence, ROC-AUC is flattered by the huge pool of easy true negatives; PR-AUC only rewards catching the rare fraud cases. |
| ROC-AUC | 0.852 | Reported for completeness, not as the headline. |
| Brier score | 0.026 | Low, meaning the calibrated probability is close to the true fraud rate at that score. |
| Expected Calibration Error | 0.005 | Confirms the isotonic calibration step did its job; the Governor's rupee math depends on this being small. |

Operating points on the held-out fold:

| Threshold | Precision | Recall | F1 | False positive rate |
|---|---|---|---|---|
| 0.10 (high recall) | 23.8% | 50.3% | 32.3% | 5.82% |
| 0.50 (textbook default) | 75.0% | 22.3% | 34.4% | 0.27% |
| 0.90 (high precision) | 81.6% | 3.3% | 6.4% | 0.03% |

## Ablation: does the graph actually earn its place

Recall at a fixed 1% false-positive rate, same estimator and chronological fold, only the feature family changes:

| Feature family | Recall at 1% FPR |
|---|---|
| Intrinsic only | 35.97% |
| + velocity | 34.93% |
| + velocity + graph | 34.64% |

Said plainly: in this run, adding velocity and graph features did not improve recall at this specific fixed-FPR cut over intrinsic features alone. We are reporting that as measured rather than hiding it, because a model whose author quietly drops an unflattering ablation is exactly the kind of thing this track is meant to catch. The graph features still matter for the ring-inspector view and for the component fraud-rate signal used elsewhere, but on this narrow slice of the curve they did not move recall.

## The cost curve and the rupee claim

`src/costs.py` names every assumption: false-negative cost is the transaction amount plus a fixed dispute-handling cost and a chargeback fee, false-positive cost is a fraction of ticket value for lost merchant margin plus expected churn friction, and review cost is a fixed per-transaction analyst cost. `src/evaluate.py` sweeps the threshold from 0 to 1 and writes the full curve to `reports/cost_curve.csv` and `reports/cost_curve.png`.

Same model, same held-out fold, only the cutoff changes:

- Fixed 0.50 threshold, the textbook default: **saves about 4,81,644 rupees per 10,000 transactions**, net of false-positive friction, against an allow-everything baseline.
- Governor's expected-value optimum, threshold 0.22: **saves 8,18,133 rupees per 10,000 transactions**.

That is roughly 3,36,000 more rupees saved per 10,000 transactions, purely from choosing the cutoff by expected value instead of a generic default. This comparison runs live in the console under "The thesis, same model, different cutoff," computed from `reports/metrics.json` and `reports/cost_curve.csv` with no numbers hand-typed into the page.

### Sensitivity analysis

The friction-cost assumption is exactly that, an assumption. Re-running the sweep at half and double the assumed false-positive friction rate:

| Friction multiplier | Optimal threshold | Rupees saved per 10,000 |
|---|---|---|
| 0.5x | 0.22 | 8,53,233 |
| 1x (assumed) | 0.22 | 8,18,133 |
| 2x | 0.22 | 7,47,934 |

The optimal threshold does not move even when the cost assumption is wrong by a factor of two in either direction, and the savings estimate only shifts by about 13% across that range. Showing where a conclusion would break is more convincing than only showing that it currently holds.

The IEEE dataset's amount field is unitless and USD-like. Every rupee figure above applies the explicit `AMOUNT_TO_INR = 83` multiplier defined in `src/config.py`, so the INR framing is a documented transformation of the source data, not a fabricated currency label.

## Reproduce it

```bash
pip install -r requirements.txt
python data/download.py
python -m src.train
```

`data/download.py` needs Kaggle API credentials (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`), and the account must have accepted the competition rules for `ieee-fraud-detection` on Kaggle's website first. The Kaggle API returns a 403 for the download until that manual step is done; there is no way to accept the rules through the API itself. Without credentials, `python -m src.train` falls back to a clearly labelled synthetic demo dataset so the pipeline and API still run end to end, and every number produced that way is tagged `synthetic-demo-fallback`, never presented as a benchmark result.

Then serve the console:

```bash
uvicorn api.main:app --reload
```

Open http://localhost:8000. The console is one static HTML file, has a synthetic INR feed marker wherever the data is illustrative, and reads live from the API for everything else. `POST /v1/score`, `POST /v1/decide`, `GET /v1/monitor/spikes`, `GET /v1/monitor/drift`, `POST /v1/advocate/packet`, `GET /v1/metrics`, and `GET /v1/metrics/cost-curve` are documented by FastAPI's OpenAPI schema at `/docs`.

## The mark

The Prahari mark is a hexagonal aperture holding the same three-node hub-and-spoke motif used in the console's own ring inspector, one coral anchor node radiating to three amber artifact nodes at exactly 120 degrees apart. It is a direct quotation of the product's real UI rather than a generic shield or padlock, because the actual differentiator here is correlating shared identity signals, not simply blocking. Source files live at `web/assets/mark.svg` and `web/assets/lockup.svg`; both are outline-only so they drop onto any background without a separate light and dark export.

## Honest limitations

- **Reject inference.** Labels are only observed for transactions that were approved in the original system, so precision here is conditional on that earlier filter and would likely look different against transactions that system declined outright.
- **Concept drift.** A single chronological window, 183 days here, cannot promise that fraud patterns stay stable beyond it. The PSI drift monitor exists to catch when the training window has gone stale, not to prevent drift from happening.
- **Component fraud rate depends on label availability.** The graph feature that scores a component's historical fraud rate is target-derived and only valid because it is computed with strict expanding-window logic; in production this feature is only as good as how quickly confirmed fraud labels arrive.
- **Cost assumptions are assumptions.** Dispute fees, margin loss, friction, and review cost in `src/config.py` are reasonable estimates, not measurements from a live merchant. The sensitivity analysis above exists specifically so this weakness is visible rather than buried.
- Synthetic demo records animate the console only and never enter the metrics reported above.

## Defense-only and privacy

> Prahari is detection, decisioning, and dispute-response only. It contains no capability to generate, simulate, obfuscate, or optimise fraudulent transactions, no adversarial evasion tooling, and no synthetic-identity generation. The synthetic demo stream produces innocuous display records for the console and is incapable of representing or testing evasion strategies.

No personally identifiable information is stored. IEEE-CIS is already anonymised, raw data is never committed, and model artifacts contain no re-identifiable information.

## Development

`make test`, `make train`, `make serve`, and `docker compose up --build` are all supported. The test suite verifies causal velocity and graph features (no row can see the future), expected-value cost math, Advocate's refusal of any claim it cannot back with real evidence, and the API's response contracts.
