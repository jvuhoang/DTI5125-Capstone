"""
symptom_scorer.py — Ensemble Disease Probability Scorer
========================================================
Scores a filled ClinicalTemplate against all three trained classifiers
(LinearSVC/LogisticRegression + BioBERT) and returns an ensemble
probability distribution over the 6 disease classes.

The ensemble averages the softmax probabilities from the TF-IDF classifier
and BioBERT. If BioBERT is not available, falls back to the sklearn model only.

Results are displayed in Streamlit as coloured progress bars with a
mandatory medical disclaimer.
"""

import numpy as np
import joblib
import os
from typing import Optional

MODEL_DIR = "biobert_classifier"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_sklearn_models():
    """Load the saved TF-IDF classifier, vectorizer, and label encoder."""
    required = ["disease_classifier.pkl", "tfidf_vectorizer.pkl", "label_encoder.pkl"]
    for f in required:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Required file not found: {f}\n"
                "Run phase3_ml.py first to train and save the classifiers."
            )
    clf        = joblib.load("disease_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    le         = joblib.load("label_encoder.pkl")
    return clf, vectorizer, le


def load_biobert_model():
    """
    Load the fine-tuned BioBERT classifier.
    Returns (tokenizer, model) or (None, None) if not available.

    Set environment variable NORA_SKIP_BIOBERT=1 to skip loading entirely
    (useful when transformers causes import crashes on some platforms).
    """
    if os.environ.get("NORA_SKIP_BIOBERT", "0") == "1":
        print("[INFO] NORA_SKIP_BIOBERT=1 — skipping BioBERT, using sklearn only.")
        return None, None

    if not os.path.exists(MODEL_DIR):
        return None, None

    # Prevent Apple Silicon / multiprocessing segfaults from tokenizers
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"[WARN] Could not load BioBERT: {e}. Falling back to sklearn only.")
        return None, None


def load_all_models():
    """Load all scorer models. Returns a dict of loaded components."""
    clf, vectorizer, le = load_sklearn_models()
    tok_bert, mod_bert  = load_biobert_model()
    return {
        "clf":        clf,
        "vectorizer": vectorizer,
        "le":         le,
        "tok_bert":   tok_bert,
        "mod_bert":   mod_bert,
    }


# ── Probability computation ───────────────────────────────────────────────────

def _sklearn_probabilities(text: str, clf, vectorizer) -> np.ndarray:
    """
    Get softmax probability distribution from the sklearn classifier.
    LinearSVC uses decision_function → softmax; LogisticRegression uses predict_proba.
    """
    vec = vectorizer.transform([text])

    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(vec)[0]

    # LinearSVC: convert decision_function to probabilities via softmax
    scores = clf.decision_function(vec)[0]
    scores = scores - scores.max()          # numerical stability
    exp_s  = np.exp(scores)
    return exp_s / exp_s.sum()


def _biobert_probabilities(text: str, tokenizer, model) -> Optional[np.ndarray]:
    """Get softmax probability distribution from fine-tuned BioBERT."""
    try:
        import torch
        enc = tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**enc).logits[0].numpy()

        logits = logits - logits.max()
        exp_l  = np.exp(logits)
        return exp_l / exp_l.sum()
    except Exception as e:
        print(f"[WARN] BioBERT inference failed: {e}")
        return None


def score_template(template_text: str, models: dict) -> dict:
    """
    Compute ensemble disease probability for a clinical template text string.

    Parameters
    ----------
    template_text : output of ClinicalTemplate.to_text()
    models        : dict returned by load_all_models()

    Returns
    -------
    dict mapping disease label → probability (float, sums to ~1.0)
    """
    clf        = models["clf"]
    vectorizer = models["vectorizer"]
    le         = models["le"]
    tok_bert   = models["tok_bert"]
    mod_bert   = models["mod_bert"]

    proba_sklearn = _sklearn_probabilities(template_text, clf, vectorizer)

    if tok_bert is not None and mod_bert is not None:
        proba_bert = _biobert_probabilities(template_text, tok_bert, mod_bert)
        if proba_bert is not None:
            ensemble = (proba_sklearn + proba_bert) / 2.0
        else:
            ensemble = proba_sklearn
    else:
        ensemble = proba_sklearn

    return {
        disease: float(prob)
        for disease, prob in zip(le.classes_, ensemble)
    }


# ── Confidence label ──────────────────────────────────────────────────────────

def confidence_label(prob: float, severity: Optional[str] = None) -> str:
    """
    Produce a confidence label for display.
    Severity from the template can modulate the label.
    """
    if prob >= 0.60:
        base = "High"
    elif prob >= 0.30:
        base = "Moderate"
    else:
        base = "Low"

    if severity == "severe" and prob >= 0.40:
        return f"{base} (severe symptoms reported)"
    return base


# ── Text-based score display (chat message) ──────────────────────────────────

def format_text_scores(scores: dict, template=None) -> str:
    """
    Return a markdown string with █░ probability bars for use in the
    chat message body (not the Streamlit panel).

    Each disease gets a 20-block bar where filled blocks (█) are
    proportional to its probability percentage.

    Parameters
    ----------
    scores   : dict mapping disease label → probability float
    template : unused — kept for call-site compatibility
    """
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_disease, top_prob = sorted_scores[0]

    if top_prob >= 0.60:
        confidence = "high"
    elif top_prob >= 0.30:
        confidence = "medium"
    else:
        confidence = "low"

    lines = [
        "**Symptom Assessment**\n",
        "Based on the information provided, here is how the reported symptoms "
        "compare to patterns in the research literature:\n",
    ]

    # Show top 3 diseases only — avoids cluttering the chat with low-probability entries
    for disease, prob in sorted_scores[:3]:
        pct     = int(prob * 100)
        bar_len = int(pct / 5)          # 20 blocks total (each block = 5%)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"**{disease}**")
        lines.append(f"`{bar}` {pct}%\n")

    lines.append(
        "\n---\n"
        "⚕️ **Important:** This assessment is based on pattern matching against "
        "published research literature and is **not a medical diagnosis**. "
        "Please consult a qualified neurologist or healthcare professional "
        "for proper evaluation and diagnosis."
    )

    return "\n".join(lines)


# ── Streamlit rendering ───────────────────────────────────────────────────────

def render_score_panel(scores: dict, severity: Optional[str] = None) -> None:
    """
    Render the disease probability panel in Streamlit.
    Shows coloured progress bars sorted by probability, with a disclaimer.
    """
    import streamlit as st

    st.subheader("🧠 Disease Probability Assessment")

    # Sort highest probability first — show top 3 only
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

    # Colour mapping per disease for visual distinction
    bar_colors = {
        "Alzheimer":  "#E63946",
        "Parkinson":  "#457B9D",
        "ALS":        "#2A9D8F",
        "Huntington": "#E9C46A",
        "Dementia":   "#F4A261",
        "Stroke":     "#264653",
    }

    for disease, prob in sorted_scores:
        pct   = prob * 100
        label = confidence_label(prob, severity)
        color = bar_colors.get(disease, "#888888")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{disease}**")
            st.progress(prob)
        with col2:
            st.metric(label="", value=f"{pct:.1f}%", delta=label,
                      delta_color="off")

    st.divider()
    st.warning(
        "⚠️ **Important Disclaimer:** This assessment is **not a medical diagnosis**. "
        "These probability scores are generated by machine learning models trained "
        "on research literature and are intended for informational purposes only. "
        "Please consult a qualified healthcare professional for any medical concerns."
    )


# ── Quick test (standalone) ───────────────────────────────────────────────────

if __name__ == "__main__":
    models = load_all_models()
    le     = models["le"]
    print(f"Disease classes: {list(le.classes_)}")

    test_cases = [
        "Patient: 70 years old, male. Symptoms: resting tremor, rigidity, bradykinesia. "
        "Duration: 3 years. Severity: moderate. Family history: father had Parkinson's. "
        "Medications: levodopa.",

        "Patient: 65 years old, female. Symptoms: progressive memory loss, confusion, "
        "difficulty with daily tasks. Duration: 2 years. Severity: moderate. "
        "Family history: mother had dementia. Medications: donepezil.",

        "Patient: 55 years old, male. Symptoms: limb weakness, muscle atrophy, "
        "fasciculations, bulbar dysfunction. Duration: 1 year. Severity: severe. "
        "Medications: riluzole.",
    ]

    for i, text in enumerate(test_cases, 1):
        scores = score_template(text, models)
        top    = max(scores, key=scores.get)
        print(f"\nCase {i}: Top prediction = {top} ({scores[top]*100:.1f}%)")
        for disease, prob in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 30)
            print(f"  {disease:<12}: {prob*100:5.1f}%  {bar}")
