import pandas as pd
from src.features.intrinsic import build_intrinsic

def test_intrinsic_missingness_and_time():
    x=pd.DataFrame({"TransactionAmt":[10.],"TransactionDT":[3600],"P_emaildomain":[None],"R_emaildomain":["x"]})
    f=build_intrinsic(x)
    assert "log_amount" in f and f["hour"].iloc[0] == 1 and f["missing_P_emaildomain"].iloc[0] == 1
