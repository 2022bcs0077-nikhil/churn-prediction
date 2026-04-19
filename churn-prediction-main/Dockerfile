# ─────────────────────────────────────────────────
# Multi-stage Dockerfile — Churn Prediction ML API
# Stage 1: Train the model
# Stage 2: Serve the inference API
# ─────────────────────────────────────────────────

# ── Stage 1: Trainer (produces model artifact) ──
FROM python:3.11-slim AS trainer

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/ ./ml/
COPY ml/data/ ./ml/data/
COPY mlops/ ./mlops/

# Train the model; artifact lands in /app/models/
RUN python ml/train.py \
    --data-dir data \
    --model-dir models \
    --mlflow-uri sqlite:///mlflow.db \
    --model random_forest


# ── Stage 2: Inference API ──
FROM python:3.11-slim AS inference

WORKDIR /app

# Runtime dependencies only
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app/ ./app/
COPY ml/ ./ml/

# Copy trained model from trainer stage
COPY --from=trainer /app/models/ ./models/

# Environment
ENV MODEL_PATH=models/churn_model.pkl
ENV SCHEMA_PATH=models/model_schema.json
ENV HIGH_RISK_THRESHOLD=0.65
ENV MEDIUM_RISK_THRESHOLD=0.35

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.ml_inference:app", "--host", "0.0.0.0", "--port", "8000"]