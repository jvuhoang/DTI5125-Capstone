#!/bin/bash
# =============================================================================
# setup.sh — One-time environment setup for the NORA capstone project
# =============================================================================
# Run this ONCE before running the pipeline for the first time.
#
# Usage:
#   bash setup.sh
#
# What it does:
#   1. Creates a conda environment called 'nora' with Python 3.11
#   2. Installs all pip dependencies
#   3. Installs the BC5CDR biomedical NER model (scispaCy)
#
# After setup completes, build the pipeline artifacts with:
#   conda activate nora
#   python run_pipeline.py --skip-biobert
#
# Then start the chatbot with:
#   streamlit run streamlit_app.py
# =============================================================================

set -e   # exit immediately on any error

echo ""
echo "============================================================"
echo "  NORA — Environment Setup"
echo "============================================================"

# ── Step 1: Create conda environment with Python 3.11 ────────────────────────
# spaCy 3.7.4 requires Python <=3.11; Python 3.12+ is not supported.

echo ""
echo "[1/3] Creating conda environment 'nora' (Python 3.11)..."

if conda env list | grep -q "^nora "; then
    echo "  Environment 'nora' already exists — skipping creation."
else
    conda create -n nora python=3.11 -y
    echo "  Created 'nora' environment."
fi

# Resolve the pip binary inside the nora environment
NORA_PIP=$(conda run -n nora which pip)
NORA_PYTHON=$(conda run -n nora which python)

echo "  Python: $NORA_PYTHON"
echo "  Pip:    $NORA_PIP"

# ── Step 2: Install pip dependencies ─────────────────────────────────────────

echo ""
echo "[2/3] Installing pip dependencies..."

conda run -n nora pip install \
    flask>=2.3.0 \
    gunicorn>=21.2.0 \
    rdflib>=6.3.2 \
    requests>=2.31.0 \
    "scispacy==0.5.4" \
    "spacy==3.7.4" \
    "scikit-learn>=1.3.0" \
    "scipy>=1.11.0" \
    "joblib>=1.3.0" \
    "numpy>=1.24.0" \
    "pandas>=2.0.0" \
    "torch>=2.0.0" \
    "transformers>=4.40.0" \
    "datasets>=2.14.0" \
    "sentence-transformers>=2.2.2" \
    "faiss-cpu>=1.7.4" \
    networkx>=3.1 \
    "matplotlib>=3.7.0" \
    "seaborn>=0.12.0" \
    "streamlit>=1.28.0" \
    nltk

echo "  Pip dependencies installed."

# ── Step 3: Install BC5CDR biomedical NER model ───────────────────────────────
# This model is not on PyPI — must be installed directly from the AI2 S3 bucket.
# It adds DISEASE and CHEMICAL entity recognition to spaCy.

echo ""
echo "[3/3] Installing scispaCy BC5CDR biomedical NER model..."

conda run -n nora pip install \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

echo "  BC5CDR model installed."

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the environment:"
echo "       conda activate nora"
echo ""
echo "  2. Navigate to the project folder and build the pipeline:"
echo "       python run_pipeline.py --skip-biobert"
echo ""
echo "     This downloads models, collects PubMed abstracts,"
echo "     runs NER + PICOS extraction, trains classifiers,"
echo "     and builds the FAISS vector index."
echo "     Estimated time: 60–90 minutes (first run)."
echo ""
echo "  3. Start the chatbot:"
echo "       streamlit run streamlit_app.py"
echo ""
