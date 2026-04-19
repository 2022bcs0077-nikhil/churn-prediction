"""
Feature Engineering Pipeline for Churn Prediction
Produces reproducible features for both training and inference.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. Ticket-based feature extraction
# ──────────────────────────────────────────────

def compute_ticket_features(tickets_df: pd.DataFrame, reference_date: datetime = None) -> pd.DataFrame:
    """
    Given a DataFrame of support tickets with columns:
        customer_id, created_at (datetime), category, sentiment_score
    Returns a per-customer feature DataFrame.
    """
    if reference_date is None:
        reference_date = datetime.utcnow()

    tickets_df = tickets_df.copy()
    tickets_df["created_at"] = pd.to_datetime(tickets_df["created_at"])

    features = []
    for cid, grp in tickets_df.groupby("customer_id"):
        grp = grp.sort_values("created_at")
        row = {"customer_id": cid}

        # Ticket frequency windows
        for days, label in [(7, "7d"), (30, "30d"), (90, "90d")]:
            cutoff = reference_date - timedelta(days=days)
            row[f"ticket_count_{label}"] = int((grp["created_at"] >= cutoff).sum())

        # Average sentiment score (lower = more negative)
        row["avg_sentiment_score"] = float(grp["sentiment_score"].mean()) if "sentiment_score" in grp else 0.0

        # Category counts (complaint, billing, technical, other)
        if "category" in grp.columns:
            for cat in ["complaint", "billing", "technical", "other"]:
                row[f"ticket_cat_{cat}"] = int((grp["category"].str.lower() == cat).sum())
        else:
            for cat in ["complaint", "billing", "technical", "other"]:
                row[f"ticket_cat_{cat}"] = 0

        # Mean time between tickets (days)
        if len(grp) > 1:
            deltas = grp["created_at"].diff().dropna().dt.total_seconds() / 86400
            row["mean_days_between_tickets"] = float(deltas.mean())
        else:
            row["mean_days_between_tickets"] = -1.0  # sentinel: only 1 ticket

        features.append(row)

    return pd.DataFrame(features)


# ──────────────────────────────────────────────
# 2. Customer-level feature extraction
# ──────────────────────────────────────────────

def compute_customer_features(customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a Telco customer DataFrame, returns engineered features.
    Expects columns: customer_id, monthly_charges, tenure, contract,
                     internet_service, payment_method, etc.
    """
    df = customers_df.copy()

    # Encode contract type
    contract_map = {"Month-to-Month": 0, "One year": 1, "Two year": 2}
    df["contract_encoded"] = df["contract"].map(contract_map).fillna(0).astype(int)

    # Encode internet service
    internet_map = {"No": 0, "DSL": 1, "Fiber optic": 2}
    df["internet_encoded"] = df.get("internet_service", pd.Series(["No"] * len(df))).map(internet_map).fillna(0).astype(int)

    # Monthly charge change (requires prior_monthly_charges column if available)
    if "prior_monthly_charges" in df.columns:
        df["monthly_charge_change"] = df["monthly_charges"] - df["prior_monthly_charges"]
    else:
        df["monthly_charge_change"] = 0.0

    # Normalize tenure into segments
    df["tenure_bucket"] = pd.cut(
        df.get("tenure", pd.Series([0] * len(df))),
        bins=[-1, 12, 24, 48, float("inf")],
        labels=[0, 1, 2, 3]
    ).astype(int)

    keep_cols = [
        "customer_id",
        "monthly_charges",
        "monthly_charge_change",
        "contract_encoded",
        "internet_encoded",
        "tenure_bucket",
    ]
    # Keep only columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols]


# ──────────────────────────────────────────────
# 3. Merge and return final feature matrix
# ──────────────────────────────────────────────

FEATURE_COLUMNS = [
    "monthly_charges",
    "monthly_charge_change",
    "contract_encoded",
    "internet_encoded",
    "tenure_bucket",
    "ticket_count_7d",
    "ticket_count_30d",
    "ticket_count_90d",
    "avg_sentiment_score",
    "ticket_cat_complaint",
    "ticket_cat_billing",
    "ticket_cat_technical",
    "ticket_cat_other",
    "mean_days_between_tickets",
]


def build_feature_matrix(
    customers_df: pd.DataFrame,
    tickets_df: pd.DataFrame,
    reference_date: datetime = None,
) -> pd.DataFrame:
    """
    Merges customer + ticket features into a single matrix.
    Returns DataFrame with FEATURE_COLUMNS + customer_id.
    """
    cust_feat = compute_customer_features(customers_df)
    ticket_feat = compute_ticket_features(tickets_df, reference_date)

    if ticket_feat.empty or "customer_id" not in ticket_feat.columns:
        merged = cust_feat.copy()
    else:
        merged = cust_feat.merge(ticket_feat, on="customer_id", how="left")

    # Ensure ticket columns exist (customers with no tickets)
    ticket_fill = {
        "ticket_count_7d": 0,
        "ticket_count_30d": 0,
        "ticket_count_90d": 0,
        "avg_sentiment_score": 0.0,
        "ticket_cat_complaint": 0,
        "ticket_cat_billing": 0,
        "ticket_cat_technical": 0,
        "ticket_cat_other": 0,
        "mean_days_between_tickets": -1.0,
    }
    for col, default in ticket_fill.items():
        if col not in merged.columns:
            merged[col] = default
    merged = merged.fillna(ticket_fill)

    return merged


# ──────────────────────────────────────────────
# 4. Sklearn-compatible transformer (for Pipeline)
# ──────────────────────────────────────────────

class ChurnFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Wraps build_feature_matrix for use inside an sklearn Pipeline.
    Expects input X to be a dict or DataFrame with keys:
        'customers' and 'tickets'
    """

    def __init__(self, reference_date=None):
        self.reference_date = reference_date
        self.feature_columns_ = FEATURE_COLUMNS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, dict):
            customers_df = X["customers"]
            tickets_df = X["tickets"]
        else:
            raise ValueError("Input must be a dict with 'customers' and 'tickets' keys.")

        matrix = build_feature_matrix(customers_df, tickets_df, self.reference_date)

        # Ensure all expected columns present
        for col in FEATURE_COLUMNS:
            if col not in matrix.columns:
                matrix[col] = 0.0

        return matrix[FEATURE_COLUMNS].values.astype(float)


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    import random

    random.seed(42)
    np.random.seed(42)

    n_customers = 100
    customers = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n_customers)],
        "monthly_charges": np.random.uniform(20, 120, n_customers),
        "prior_monthly_charges": np.random.uniform(20, 120, n_customers),
        "tenure": np.random.randint(1, 72, n_customers),
        "contract": np.random.choice(["Month-to-Month", "One year", "Two year"], n_customers),
        "internet_service": np.random.choice(["No", "DSL", "Fiber optic"], n_customers),
    })

    ticket_rows = []
    base = datetime.utcnow()
    for cid in customers["customer_id"]:
        for _ in range(random.randint(0, 8)):
            ticket_rows.append({
                "customer_id": cid,
                "created_at": base - timedelta(days=random.randint(0, 90)),
                "category": random.choice(["complaint", "billing", "technical", "other"]),
                "sentiment_score": random.uniform(-1, 1),
            })

    tickets = pd.DataFrame(ticket_rows)
    matrix = build_feature_matrix(customers, tickets)
    print("Feature matrix shape:", matrix.shape)
    print(matrix[FEATURE_COLUMNS].head())