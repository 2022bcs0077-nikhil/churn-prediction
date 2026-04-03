import pandas as pd
import numpy as np
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset
import json
import os
from datetime import datetime

def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: str = "mlops/drift/reports"
) -> dict:
    print("🔍 Running data drift detection...")

    # Select only numeric columns
    num_cols = reference_data.select_dtypes(include=[np.number]).columns.tolist()
    ref = reference_data[num_cols].reset_index(drop=True)
    cur = current_data[num_cols].reset_index(drop=True)

    # Define data schema
    data_def = DataDefinition()

    # Create Evidently datasets
    ref_dataset = Dataset.from_pandas(ref, data_definition=data_def)
    cur_dataset = Dataset.from_pandas(cur, data_definition=data_def)

    # Run drift report
    report = Report([DataDriftPreset()])
    run = report.run(reference_data=ref_dataset, current_data=cur_dataset)

    # Save HTML report
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{output_path}/drift_report_{timestamp}.html"
    run.save_html(report_path)
    print(f"📄 Drift report saved to {report_path}")

    # Extract summary from dict
    report_dict = run.dict()
    metrics = report_dict.get("metrics", [])

    drifted_cols = 0
    drift_share = 0.0
    drift_detected = False

    for metric in metrics:
        result = metric.get("result", {})
        if "drifted_columns" in result:
            drifted_cols = result["drifted_columns"]
        if "share_of_drifted_columns" in result:
            drift_share = result["share_of_drifted_columns"]
        if "dataset_drift" in result:
            drift_detected = result["dataset_drift"]

    drift_summary = {
        "timestamp": timestamp,
        "dataset_drift_detected": drift_detected,
        "drift_share": drift_share,
        "number_of_drifted_columns": drifted_cols,
        "report_path": report_path
    }

    print(f"✅ Drift detected:  {drift_summary['dataset_drift_detected']}")
    print(f"✅ Drift share:     {drift_summary['drift_share']:.2%}")
    print(f"✅ Drifted columns: {drift_summary['number_of_drifted_columns']}")

    return drift_summary

def simulate_production_drift(reference_data: pd.DataFrame) -> pd.DataFrame:
    current_data = reference_data.copy()
    np.random.seed(99)

    numerical_cols = current_data.select_dtypes(include=[np.number]).columns

    for col in numerical_cols[:5]:
        current_data[col] = (
            current_data[col] * np.random.uniform(1.1, 1.3)
            + np.random.normal(0, current_data[col].std() * 0.2, len(current_data))
        )

    return current_data

if __name__ == "__main__":
    from ml.features import load_and_preprocess, simulate_ticket_features

    df = load_and_preprocess("ml/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = simulate_ticket_features(df)

    reference = df.sample(1000, random_state=42)
    current = simulate_production_drift(df.sample(1000, random_state=99))

    ref_features = reference.drop(columns=["Churn"])
    cur_features = current.drop(columns=["Churn"])

    summary = detect_data_drift(ref_features, cur_features)
    print("\n📊 Drift Summary:")
    print(json.dumps(summary, indent=2, default=str))
