from pydantic import BaseModel
from typing import List
from datetime import datetime

class Ticket(BaseModel):
    ticket_id: str
    created_at: datetime
    category: str  # e.g., "complaint", "billing", "technical"
    description: str

class CustomerRequest(BaseModel):
    customer_id: str
    contract_type: str  # "Month-to-Month", "One Year", "Two Year"
    monthly_charges: float
    previous_monthly_charges: float
    tickets: List[Ticket]

class RiskResponse(BaseModel):
    customer_id: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    triggered_rule: str

