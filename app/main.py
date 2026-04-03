from fastapi import FastAPI, HTTPException
from app.models import CustomerRequest, RiskResponse
from app.rules import compute_churn_risk
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Risk Prediction API",
    description="Rule-based customer churn risk prediction system",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict-risk", response_model=RiskResponse)
def predict_risk(customer: CustomerRequest):
    try:
        logger.info(f"Received request for customer_id: {customer.customer_id}")

        result = compute_churn_risk(customer)

        logger.info(
            f"Customer {customer.customer_id} → "
            f"Risk: {result['risk_level']} | "
            f"Rule: {result['triggered_rule']}"
        )

        return RiskResponse(
            customer_id=customer.customer_id,
            risk_level=result["risk_level"],
            triggered_rule=result["triggered_rule"]
        )

    except Exception as e:
        logger.error(f"Error processing customer {customer.customer_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
