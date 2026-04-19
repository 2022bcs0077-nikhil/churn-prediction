"""
Drift Detection Module (MLOps Layer)
Uses Evidently to detect:
  - Feature drift (data distribution shift)
  - Concept drift (label distribution shift)
  - Prediction drift (model output shift)

Run: python mlops/drift_detection.py
"""

import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.feature_engineering import build_feature_matrix, FEATURE_COLUMNS
from ml.train import load_or_simulate_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────
# Evidently-based drift report
# ──────────────────────────────────────────────

def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    out_dir: str = "drift_reports",
    label: str = None,
):
    """
    Generate an Evidently data drift report comparing reference vs current data.
    Returns dict with drift summary.
    """
    os.makedirs(out_dir, exist_ok=True)
    label = label or datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        report.run(reference_data=reference_df, current_data=current_df)

        html_path = os.path.join(out_dir, f"drift_report_{label}.html")
        report.save_html(html_path)
        logger.info("Drift report saved to %s", html_path)

        # Extract drift summary
        result = report.as_dict()
        drift_metrics = result.get("metrics", [])
        dataset_drift = next(
            (m for m in drift_metrics if m.get("metric") == "DatasetDriftMetric"), {}
        )
        drift_detected = dataset_drift.get("result", {}).get("dataset_drift", False)
        drift_share = dataset_drift.get("result", {}).get("share_of_drifted_columns", 0.0)

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_detected": drift_detected,
            "drift_share": drift_share,
            "report_path": html_path,
        }

    except ImportError:
        logger.warning("Evidently not installed — using statistical fallback (KS test)")
        summary = _fallback_drift_detection(reference_df, current_df)
        summary["report_path"] = None

    return summary


def _fallback_drift_detection(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """
    Statistical fallback: Kolmogorov-Smirnov test per feature.
    Returns summary without needing Evidently.
    """
    from scipy import stats

    drifted_cols = []
    p_values = {}

    for col in FEATURE_COLUMNS:
        if col in reference_df.columns and col in current_df.columns:
            ks_stat, p_val = stats.ks_2samp(
                reference_df[col].dropna().values,
                current_df[col].dropna().values
            )
            p_values[col] = round(float(p_val), 4)
            if p_val < 0.05:
                drifted_cols.append(col)

    drift_share = len(drifted_cols) / len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 0.0

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "drift_detected": drift_share > 0.2,  # Alert if >20% of features drift
        "drift_share": round(drift_share, 4),
        "drifted_columns": drifted_cols,
        "p_values": p_values,
        "method": "KS-test fallback",
    }


# ──────────────────────────────────────────────
# Prediction drift (concept drift proxy)
# ──────────────────────────────────────────────

def check_prediction_drift(
    model_path: str,
    reference_features: np.ndarray,
    current_features: np.ndarray,
    threshold: float = 0.1,
) -> dict:
    """
    Check if model output distribution has shifted significantly.
    Uses PSI (Population Stability Index) on predicted probabilities.
    """
    model = joblib.load(model_path)

    ref_probs = model.predict_proba(reference_features)[:, 1]
    cur_probs = model.predict_proba(current_features)[:, 1]

    psi = _compute_psi(ref_probs, cur_probs)

    return {
        "psi": round(float(psi), 4),
        "drift_detected": psi > threshold,
        "severity": "high" if psi > 0.25 else ("medium" if psi > 0.10 else "low"),
        "ref_mean_prob": round(float(ref_probs.mean()), 4),
        "cur_mean_prob": round(float(cur_probs.mean()), 4),
    }


def _compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index."""
    bins = np.linspace(0, 1, buckets + 1)
    bins[0] = -np.inf
    bins[-1] = np.inf

    exp_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    act_pct = np.histogram(actual, bins=bins)[0] / len(actual)

    # Avoid zero division
    exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


# ──────────────────────────────────────────────
# Monitoring alert
# ──────────────────────────────────────────────

def alert_if_drift(summary: dict, out_dir: str = "drift_reports"):
    """Log an alert and write a marker file if drift is detected."""
    if summary.get("drift_detected"):
        logger.warning("🚨 DATA DRIFT DETECTED — share=%.2f", summary.get("drift_share", 0))
        alert_path = os.path.join(out_dir, "DRIFT_ALERT.json")
        with open(alert_path, "w") as f:
            json.dump(summary, f, indent=2)
        return True
    else:
        logger.info("✅ No significant drift detected (share=%.2f)", summary.get("drift_share", 0))
        return False


# ──────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="drift_reports")
    parser.add_argument("--model-path", default="models/churn_model.pkl")
    args = parser.parse_args()

    logger.info("Loading reference & current data...")
    # In production: reference = training data snapshot; current = latest batch
    # Here we simulate both from the same generator with different seeds
    np.random.seed(0)
    customers_ref, tickets_ref, labels_ref = load_or_simulate_data()
    ref_df = build_feature_matrix(customers_ref, tickets_ref)[FEATURE_COLUMNS]

    np.random.seed(99)  # Different seed = simulates distribution shift
    customers_cur, tickets_cur, labels_cur = load_or_simulate_data()
    cur_df = build_feature_matrix(customers_cur, tickets_cur)[FEATURE_COLUMNS]

    logger.info("Running drift report...")
    summary = run_drift_report(ref_df, cur_df, out_dir=args.out_dir)
    alert_if_drift(summary, out_dir=args.out_dir)

    print("\nDrift Summary:")
    print(json.dumps(summary, indent=2))