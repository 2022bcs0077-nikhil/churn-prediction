from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime


class Ticket(BaseModel):
    ticket_id: str
    created_at: datetime
    category: str
    description: str


class CustomerRequest(BaseModel):
    customer_id: str
    contract_type: str
    monthly_charges: float
    previous_monthly_charges: float
    tickets: List[Ticket]


class RiskResponse(BaseModel):
    customer_id: str
    risk_level: str
    triggered_rule: str


class MLPredictRequest(BaseModel):
    customer_id: str
    features: Dict[str, Any]


class MLPredictResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_level: str