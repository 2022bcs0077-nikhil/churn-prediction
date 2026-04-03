import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
from ml.features import load_and_preprocess, simulate_ticket_features, get_feature_columns

def train_model(data_path: str, model_output_path: str):
    print("📦 Loading and preprocessing data...")
    df = load_and_preprocess(data_path)

    print("🎫 Simulating ticket features...")
    df = simulate_ticket_features(df)

    # Split features and target
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["Churn"]

    print(f"✅ Dataset shape: {X.shape}")
    print(f"✅ Churn distribution:\n{y.value_counts()}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🤖 Training Random Forest Classifier...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Save model and feature columns
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "feature_columns": feature_cols
    }, model_output_path)

    print(f"💾 Model saved to {model_output_path}")

    return pipeline, X_test, y_test

if __name__ == "__main__":
    train_model(
        data_path="ml/data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        model_output_path="ml/models/model.pkl"
    )
