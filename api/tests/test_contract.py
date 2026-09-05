from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health"); assert r.status_code == 200 and r.json()["status"] == "ok"

def test_score_contract():
    r = client.post("/v1/score", json={"amount": 1200, "merchant_category": "fashion"})
    assert r.status_code == 200
    body = r.json()
    assert {"score","calibrated_probability","action","ring","explanations","model_version"} <= body.keys()

def test_decide_contract():
    r = client.post("/v1/decide", json={"amount": 1200})
    assert r.status_code == 200
    assert len(r.json()["expected_values"]) == 4
