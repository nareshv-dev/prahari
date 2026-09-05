from typing import Any
from pydantic import BaseModel, Field

class ScoreRequest(BaseModel):
    transaction_id: str = "demo-00001"
    amount: float = Field(1000, gt=0)
    merchant_category: str = "electronics"
    features: dict[str, Any] = Field(default_factory=dict)

class Ring(BaseModel):
    in_ring: bool = False
    component_size: int = 0
    ring_score: float = 0.0

class ScoreResponse(BaseModel):
    score: float
    calibrated_probability: float
    action: str
    threshold_used: float
    segment: str
    ring: Ring
    explanations: list[dict[str, Any]]
    model_version: str

class DecideResponse(ScoreResponse):
    expected_values: dict[str, float]

class AdvocateRequest(BaseModel):
    transaction_id: str
