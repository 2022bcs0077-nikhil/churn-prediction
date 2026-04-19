"""
Model Evaluation Script
Generates F1, ROC-AUC, Precision-Recall reports and plots.
Run standalone: python ml/evaluate.py --model-path models/churn_model.pkl
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
import random

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
import mlflow

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.train import load_or_simulate_data
from ml.feature_engineering import build_feature_matrix, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_roc_curve(y_test, y_prob, out_dir):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Churn Prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_precision_recall(y_test, y_prob, out_dir):
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, lw=2, label=f"PR curve (AP = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Churn Prediction")
    plt.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(out_dir, "pr_curve.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_confusion_matrix(y_test, y_pred, out_dir):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_f1_by_threshold(y_test, y_prob, out_dir):
    """Show how F1 varies with decision threshold — useful for selecting optimal cutoff."""
    thresholds = np.linspace(0.1, 0.9, 81)
    f1s = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    best_t = thresholds[np.argmax(f1s)]
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, f1s, lw=2)
    plt.axvline(best_t, linestyle="--", color="red", label=f"Best threshold = {best_t:.2f}")
    plt.xlabel("Decision Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Decision Threshold")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "f1_threshold.png")
    plt.savefig(path, dpi=120)
    plt.close()
    return path, best_t


def evaluate(model_path: str, data_dir: str = "data", out_dir: str = "eval_output",
             mlflow_uri: str = "sqlite:///mlflow.db", log_to_mlflow: bool = True):
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Loading model from %s", model_path)
    pipeline = joblib.load(model_path)

    logger.info("Loading evaluation data...")
    customers_df, tickets_df, labels = load_or_simulate_data(data_dir)
    feature_matrix = build_feature_matrix(customers_df, tickets_df)
    X = feature_matrix[FEATURE_COLUMNS].values.astype(float)
    y = labels.values

    # Use 20% held-out split (same seed as training to respect split)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # ── Metrics ──
    metrics = {
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
    }

    report_dict = classification_report(y_test, y_pred, target_names=["Not Churn", "Churn"],
                                        output_dict=True)
    metrics.update({
        "precision_churn": report_dict["Churn"]["precision"],
        "recall_churn": report_dict["Churn"]["recall"],
        "precision_not_churn": report_dict["Not Churn"]["precision"],
        "recall_not_churn": report_dict["Not Churn"]["recall"],
    })

    # ── Plots ──
    roc_path = plot_roc_curve(y_test, y_prob, out_dir)
    pr_path = plot_precision_recall(y_test, y_prob, out_dir)
    cm_path = plot_confusion_matrix(y_test, y_pred, out_dir)
    f1t_path, best_threshold = plot_f1_by_threshold(y_test, y_prob, out_dir)
    metrics["best_threshold"] = best_threshold

    # ── Save report ──
    report_str = classification_report(y_test, y_pred, target_names=["Not Churn", "Churn"])
    report_path = os.path.join(out_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Evaluation Report — {datetime.utcnow().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_str)
        f.write(f"\nROC-AUC:            {metrics['roc_auc']:.4f}\n")
        f.write(f"Average Precision:  {metrics['avg_precision']:.4f}\n")
        f.write(f"F1 Score:           {metrics['f1']:.4f}\n")
        f.write(f"Best Threshold:     {best_threshold:.2f}\n")

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Log to MLflow ──
    if log_to_mlflow:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("churn-prediction-evaluation")
        with mlflow.start_run(run_name=f"eval-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"):
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            for p in [roc_path, pr_path, cm_path, f1t_path, report_path, metrics_path]:
                mlflow.log_artifact(p)

    # ── Print summary ──
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(report_str)
    print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"Average Precision:  {metrics['avg_precision']:.4f}")
    print(f"F1 Score:           {metrics['f1']:.4f}")
    print(f"Best Threshold:     {best_threshold:.2f}")
    print(f"\nPlots saved to: {out_dir}/")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/churn_model.pkl")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="eval_output")
    parser.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        mlflow_uri=args.mlflow_uri,
        log_to_mlflow=not args.no_mlflow,
    )