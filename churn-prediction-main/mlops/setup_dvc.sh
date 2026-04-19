#!/usr/bin/env bash
# DVC Setup Script for Assignment 2 MLOps
# Run once after cloning: bash mlops/setup_dvc.sh

set -e

echo "=== Setting up DVC for data versioning ==="

# ── Install DVC ──────────────────────────────────────────────────────────────
install_dvc() {
    echo "Installing DVC..."

    # --prefer-binary avoids legacy setup.py builds (fixes voluptuous error)
    # Try methods in order until one succeeds
    pip install dvc --prefer-binary -q 2>/dev/null && return
    pip install dvc --prefer-binary --break-system-packages -q 2>/dev/null && return
    pip install dvc --prefer-binary --user -q 2>/dev/null && return

    # Last resort: create a venv
    echo "Falling back to virtual environment..."
    python3 -m venv .dvc-env
    .dvc-env/bin/pip install dvc --prefer-binary -q
    # Make dvc available on PATH for this session
    export PATH="$PWD/.dvc-env/bin:$PATH"
}

if ! command -v dvc &> /dev/null; then
    install_dvc
fi

echo "DVC version: $(dvc --version)"

# ── Init DVC ─────────────────────────────────────────────────────────────────
# --no-scm: don't assume git (safe if git already initialised)
dvc init --no-scm 2>/dev/null || true

mkdir -p data/raw data/processed data/splits models eval_output
touch data/raw/.gitkeep data/processed/.gitkeep data/splits/.gitkeep

echo "=== DVC initialized ==="

# ── dvc.yaml pipeline ────────────────────────────────────────────────────────
cat > dvc.yaml << 'EOF'
stages:
  train:
    cmd: python ml/train.py --data-dir data --model-dir models
    deps:
      - ml/train.py
      - ml/feature_engineering.py
      - data/raw
    outs:
      - models/churn_model.pkl
      - models/model_schema.json
    metrics:
      - eval_output/metrics.json:
          cache: false

  evaluate:
    cmd: python ml/evaluate.py --model-path models/churn_model.pkl --out-dir eval_output --no-mlflow
    deps:
      - ml/evaluate.py
      - models/churn_model.pkl
    metrics:
      - eval_output/metrics.json:
          cache: false
    plots:
      - eval_output/roc_curve.png
      - eval_output/pr_curve.png
      - eval_output/confusion_matrix.png
EOF

echo "=== dvc.yaml created ==="

# ── .dvcignore ───────────────────────────────────────────────────────────────
cat > .dvcignore << 'EOF'
**/__pycache__
**/*.pyc
.git
mlruns/
mlflow.db
EOF

# ── params.yaml ──────────────────────────────────────────────────────────────
cat > params.yaml << 'EOF'
train:
  model_type: random_forest
  test_size: 0.2
  random_state: 42

thresholds:
  high_risk: 0.65
  medium_risk: 0.35
  min_f1: 0.70
  drift_retrain: 0.20

features:
  ticket_windows: [7, 30, 90]
  sentiment_weight: 1.0
EOF

echo "=== params.yaml created ==="
echo ""
echo "✅ DVC setup complete!"
echo ""
echo "Next steps:"
echo "  dvc repro          # Run full pipeline (train + evaluate)"
echo "  dvc metrics show   # Show current metrics"
echo "  dvc params diff    # Compare params between commits"
echo "  dvc dag            # Visualize pipeline DAG"