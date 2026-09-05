"""Central, documented Prahari configuration."""
from pathlib import Path

SEED = 42
AMOUNT_TO_INR = 83.0  # transparent illustrative USD-like IEEE unit conversion.
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_VERSION = "demo-v1"

# Cost assumptions, expressed as fractions of the transaction value where noted.
C_FN_FIXED = 25.0       # INR dispute handling labour
C_CHARGEBACK_FEE = 20.0 # INR payment-network fee
FP_MARGIN_RATE = 0.18   # merchant contribution margin foregone
FP_FRICTION_RATE = 0.04 # expected churn/support friction
C_REVIEW = 18.0         # INR analyst review cost
REVIEW_FRACTION = 0.55  # reviewed transactions get this probability of avoiding loss

OPERATING_POINTS = {"high_recall": 0.10, "balanced": 0.50, "high_precision": 0.90}
PSI_ALERT = 0.20
SPIKE_Z_ALERT = 3.0
