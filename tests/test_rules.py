from datetime import datetime, timezone, timedelta
from app.models import CustomerRequest, Ticket
from app.rules import compute_churn_risk

def make_ticket(ticket_id, days_ago, category="general", description="Test ticket"):
    return Ticket(
        ticket_id=ticket_id,
        created_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
        category=category,
        description=description
    )

def test_high_risk_more_than_5_tickets_30d():
    customer = CustomerRequest(
        customer_id="CUST001",
        contract_type="One Year",
        monthly_charges=80.0,
        previous_monthly_charges=80.0,
        tickets=[make_ticket(f"T00{i}", days_ago=5) for i in range(6)]
    )
    result = compute_churn_risk(customer)
    assert result["risk_level"] == "HIGH"
    assert "5 tickets" in result["triggered_rule"]

def test_high_risk_month_to_month_with_complaint():
    customer = CustomerRequest(
        customer_id="CUST002",
        contract_type="Month-to-Month",
        monthly_charges=80.0,
        previous_monthly_charges=80.0,
        tickets=[make_ticket("T001", days_ago=10, category="complaint")]
    )
    result = compute_churn_risk(customer)
    assert result["risk_level"] == "HIGH"
    assert "complaint" in result["triggered_rule"]

def test_medium_risk_charge_increase_with_tickets():
    customer = CustomerRequest(
        customer_id="CUST003",
        contract_type="One Year",
        monthly_charges=90.0,
        previous_monthly_charges=70.0,
        tickets=[make_ticket(f"T00{i}", days_ago=5) for i in range(3)]
    )
    result = compute_churn_risk(customer)
    assert result["risk_level"] == "MEDIUM"
    assert "Charge increase" in result["triggered_rule"]

def test_low_risk_no_rules_triggered():
    customer = CustomerRequest(
        customer_id="CUST004",
        contract_type="Two Year",
        monthly_charges=80.0,
        previous_monthly_charges=80.0,
        tickets=[make_ticket("T001", days_ago=10, category="general")]
    )
    result = compute_churn_risk(customer)
    assert result["risk_level"] == "LOW"
