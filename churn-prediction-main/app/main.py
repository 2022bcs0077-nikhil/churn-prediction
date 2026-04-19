from fastapi import FastAPI, HTTPException
from app.models import CustomerRequest, RiskResponse, MLPredictRequest, MLPredictResponse
from app.rules import compute_churn_risk
from app.ml_model import get_predictor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Risk Prediction API",
    description="Rule-based and ML-based customer churn risk prediction system",
    version="2.0.0"
)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Stage 1: Rule-based endpoint
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


# Stage 2: ML-based endpoint
@app.post("/predict", response_model=MLPredictResponse)
def predict_churn(request: MLPredictRequest):
    try:
        logger.info(f"ML prediction request for customer_id: {request.customer_id}")
        predictor = get_predictor()
        result = predictor.predict(request.features)
        logger.info(
            f"Customer {request.customer_id} → "
            f"Churn Probability: {result['churn_probability']} | "
            f"Risk: {result['risk_level']}"
        )
        return MLPredictResponse(
            customer_id=request.customer_id,
            churn_probability=result["churn_probability"],
            churn_prediction=result["churn_prediction"],
            risk_level=result["risk_level"]
        )
    except Exception as e:
        logger.error(f"Error processing ML prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
