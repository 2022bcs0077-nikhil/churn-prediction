import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
import joblib

def evaluate_model(pipeline, X_test, y_test):
    print("\n📊 Evaluating Model...")

    # Get predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"✅ F1 Score:        {f1:.4f}")
    print(f"✅ ROC-AUC Score:   {roc_auc:.4f}")
    print(f"✅ Precision:       {precision:.4f}")
    print(f"✅ Recall:          {recall:.4f}")

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Not Churned", "Churned"]))

    print("🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return {
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4)
    }

if __name__ == "__main__":
    import joblib
    from ml.train import train_model

    pipeline, X_test, y_test = train_model(
        data_path="ml/data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        model_output_path="ml/models/model.pkl"
    )

    evaluate_model(pipeline, X_test, y_test)
