import os
import joblib
import pandas as pd
from datetime import datetime
from mlops.experiment_tracking import train_and_track
from mlops.drift.detector import detect_data_drift, simulate_production_drift
from ml.features import load_and_preprocess, simulate_ticket_features

DRIFT_THRESHOLD = 0.3  # Retrain if >30% of columns drift
DATA_PATH = "ml/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "ml/models/model.pkl"

def check_and_retrain():
    print(f"\n{'='*50}")
    print(f"🔄 Running retraining check at {datetime.now()}")
    print(f"{'='*50}")

    # Load reference data
    print("📦 Loading reference data...")
    df = load_and_preprocess(DATA_PATH)
    df = simulate_ticket_features(df)

    # Simulate production data (in real system, this would be live data)
    reference = df.sample(1000, random_state=42)
    current = simulate_production_drift(df.sample(1000, random_state=99))

    # Remove target column for drift detection
    ref_features = reference.drop(columns=["Churn"])
    cur_features = current.drop(columns=["Churn"])

    # Check for drift
    print("🔍 Checking for data drift...")
    drift_summary = detect_data_drift(ref_features, cur_features)

    drift_detected = drift_summary["dataset_drift_detected"]
    drift_share = drift_summary["drift_share"]

    print(f"\n📊 Drift Results:")
    print(f"   Drift detected: {drift_detected}")
    print(f"   Drift share:    {drift_share:.2%}")

    # Decision: retrain or not
    if drift_detected and drift_share > DRIFT_THRESHOLD:
        print(f"\n⚠️  Drift share {drift_share:.2%} exceeds threshold {DRIFT_THRESHOLD:.2%}")
        print("🚀 Triggering automated retraining...")

        pipeline, X_test, y_test, metrics = train_and_track(
            data_path=DATA_PATH,
            model_output_path=MODEL_PATH,
            experiment_name="churn-prediction-retrain"
        )
        print(f"\n✅ Retraining complete!")
        print(f"   F1 Score:  {metrics['f1_score']}")
        print(f"   ROC-AUC:   {metrics['roc_auc']}")
        return {
            "action": "retrained",
            "drift_share": drift_share,
            "metrics": metrics
        }
    else:
        print(f"\n✅ Drift share {drift_share:.2%} is within threshold")
        print("   No retraining needed")

        return {
            "action": "skipped",
            "drift_share": drift_share,
            "metrics": None
        }
if __name__ == "__main__":
    result = check_and_retrain()
    print(f"\n📋 Final Result: {result['action'].upper()}")
