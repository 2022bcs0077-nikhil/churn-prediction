from datetime import datetime, timezone
from app.models import CustomerRequest


def compute_churn_risk(customer: CustomerRequest) -> dict:
    now = datetime.now(timezone.utc)

    # Count tickets in last 7 and 30 days
    tickets_30d = [
        t for t in customer.tickets
        if (now - t.created_at.replace(tzinfo=timezone.utc)).days <= 30
    ]

    tickets_7d = [
        t for t in customer.tickets
        if (now - t.created_at.replace(tzinfo=timezone.utc)).days <= 7
    ]

    # Count complaint tickets
    complaint_tickets = [
        t for t in customer.tickets
        if t.category.lower() == "complaint"
    ]

    # Calculate charge increase percentage
    if customer.previous_monthly_charges > 0:
        charge_increase_pct = (
            (customer.monthly_charges - customer.previous_monthly_charges)
            / customer.previous_monthly_charges
        ) * 100
    else:
        charge_increase_pct = 0

    # Rule 1: HIGH - More than 5 tickets in last 30 days
    if len(tickets_30d) > 5:
        return {"risk_level": "HIGH", "triggered_rule": "More than 5 tickets in last 30 days"}

    # Rule 2: HIGH - Month-to-Month contract + complaint ticket
    if customer.contract_type == "Month-to-Month" and len(complaint_tickets) > 0:
        return {"risk_level": "HIGH", "triggered_rule": "Month-to-Month contract with complaint ticket"}

    # Rule 3: MEDIUM - Charge increased + multiple tickets
    if charge_increase_pct > 10 and len(tickets_30d) >= 3:
        return {"risk_level": "MEDIUM", "triggered_rule": "Charge increase with multiple tickets"}

    # Rule 4: MEDIUM - More than 3 tickets in last 7 days
    if len(tickets_7d) > 3:
        return {"risk_level": "MEDIUM", "triggered_rule": "More than 3 tickets in last 7 days"}

    # Default: LOW
    return {"risk_level": "LOW", "triggered_rule": "No high risk rules triggered"}
