import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import random

def load_and_preprocess(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Drop customer ID (not a feature)
    df.drop(columns=["customerID"], inplace=True)

    # Convert target column
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Fix TotalCharges (it has spaces)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode binary columns
    binary_cols = ["gender", "Partner", "Dependents",
                   "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0,
                                "Male": 1, "Female": 0})

    # Encode multi-class columns
    multi_cols = ["MultipleLines", "InternetService",
                  "OnlineSecurity", "OnlineBackup",
                  "DeviceProtection", "TechSupport",
                  "StreamingTV", "StreamingMovies",
                  "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=multi_cols)

    return df

def simulate_ticket_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate ticket behavior features since we don't have real ticket data"""
    n = len(df)
    random.seed(42)
    np.random.seed(42)

    df["ticket_count_7d"] = np.random.randint(0, 5, n)
    df["ticket_count_30d"] = np.random.randint(0, 10, n)
    df["ticket_count_90d"] = np.random.randint(0, 20, n)
    df["complaint_count"] = np.random.randint(0, 5, n)
    df["sentiment_score"] = np.random.uniform(-1, 1, n).round(2)
    df["charge_change_pct"] = (
        (df["MonthlyCharges"] - df["MonthlyCharges"].shift(1).fillna(df["MonthlyCharges"]))
        / df["MonthlyCharges"].replace(0, np.nan)
    ).fillna(0).round(2)

    # Make churned customers have worse ticket behavior
    churn_mask = df["Churn"] == 1
    df.loc[churn_mask, "ticket_count_30d"] += np.random.randint(2, 5, churn_mask.sum())
    df.loc[churn_mask, "complaint_count"] += np.random.randint(1, 3, churn_mask.sum())
    df.loc[churn_mask, "sentiment_score"] -= np.random.uniform(0.2, 0.5, churn_mask.sum())
    df["sentiment_score"] = df["sentiment_score"].clip(-1, 1)

    return df

def get_feature_columns(df: pd.DataFrame) -> list:
    return [col for col in df.columns if col != "Churn"]
