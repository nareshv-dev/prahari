import numpy as np
from src.costs import savings_curve, false_negative_cost, false_positive_cost

def test_sweep_matches_closed_form():
    y=np.array([1,0]); p=np.array([.9,.9]); a=np.array([100.,100.])
    curve=savings_curve(y,p,a,thresholds=[.5])
    expected=false_negative_cost(a[0]) - false_positive_cost(a[1])
    assert np.isclose(curve.net_saved.iloc[0], expected)
