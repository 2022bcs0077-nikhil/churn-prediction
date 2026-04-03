import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class ChurnPredictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.pipeline = None
        self.feature_columns = None
        self.load_model()

    def load_model(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        artifact = joblib.load(self.model_path)
        self.pipeline = artifact["pipeline"]
        self.feature_columns = artifact["feature_columns"]
        print(f"✅ Model loaded from {self.model_path}")

    def predict(self, features: dict) -> dict:
        # Convert to dataframe
        df = pd.DataFrame([features])

        # Add missing columns with 0
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns to match training
        df = df[self.feature_columns]

        # Get prediction and probability
        churn_prob = self.pipeline.predict_proba(df)[0][1]
        churn_pred = self.pipeline.predict(df)[0]

        return {
            "churn_probability": round(float(churn_prob), 4),
            "churn_prediction": bool(churn_pred),
            "risk_level": self._get_risk_level(churn_prob)
        }

    def _get_risk_level(self, probability: float) -> str:
        if probability >= 0.7:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

# Singleton instance
_predictor = None

def get_predictor() -> ChurnPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor("ml/models/model.pkl")
    return _predictor
