from src.advocate import representment_draft

def test_drafter_refuses_missing_claims():
    out=representment_draft({"sentinel_score": .91})
    assert out["citations"] == ["sentinel_score"]
    assert "device_fingerprint" in out["refused_fields"]
    assert "device fingerprint" not in out["draft"].lower()
