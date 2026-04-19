import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import pandas as pd
import numpy as np
from ml.features import load_and_preprocess, simulate_ticket_features, get_feature_columns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os


def train_and_track(
    data_path: str,
    model_output_path: str,
    experiment_name: str = "churn-prediction",
    n_estimators: int = 100,
    max_depth: int = 10
):
    # Set experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        print("📦 Loading and preprocessing data...")
        df = load_and_preprocess(data_path)
        df = simulate_ticket_features(df)

        feature_cols = get_feature_columns(df)
        X = df[feature_cols]
        y = df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", len(feature_cols))

        print("🤖 Training model...")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                class_weight="balanced"
            ))
        ])
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        print(f"✅ F1 Score:      {f1:.4f}")
        print(f"✅ ROC-AUC:       {roc_auc:.4f}")
        print(f"✅ Precision:     {precision:.4f}")
        print(f"✅ Recall:        {recall:.4f}")

        # Log model to MLflow
        mlflow.sklearn.log_model(pipeline, "model")

        # Save model artifact locally
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump({
            "pipeline": pipeline,
            "feature_columns": feature_cols
        }, model_output_path)

        mlflow.log_artifact(model_output_path)

        print(f"💾 Model saved and tracked with MLflow")

        return pipeline, X_test, y_test, {
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4)
        }


if __name__ == "__main__":
    train_and_track(
        data_path="ml/data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        model_output_path="ml/models/model.pkl"
    )
