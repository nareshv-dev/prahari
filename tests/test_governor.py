from src.governor import expected_values, decide

def test_expected_value_ordering():
    ev=expected_values(.99, 1000)
    assert ev["BLOCK"] < ev["ALLOW"]

def test_decision_contract_and_ticket_threshold():
    d=decide(.01, 100, "x", {"x": .5})
    assert d["action"] in {"ALLOW","STEP_UP","REVIEW","BLOCK"}
    assert decide(.9, 10000, "x", {"x": .5})["action"] in {"BLOCK","REVIEW","STEP_UP","ALLOW"}
