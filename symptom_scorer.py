"""
symptom_scorer.py — Ensemble Disease Probability Scorer
========================================================
Scores a filled ClinicalTemplate against all three trained classifiers
(LinearSVC/LogisticRegression/Random Forest + BioBERT) and returns an ensemble
probability distribution over the 5 disease classes.

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


# ── Heuristic calibration overlay ────────────────────────────────────────────
#
# The TF-IDF + SVM ensemble is trained on PubMed abstracts.  Stroke papers
# use very broad clinical vocabulary (patient demographics, family history,
# medications) which the vectoriser sees on EVERY template regardless of
# symptoms, causing a persistent Stroke over-prediction bias.
#
# This overlay applies lightweight multiplicative factors to the ensemble
# probabilities based on highly disease-specific symptom keywords found in
# the template text, then re-normalises.  Boosts are only applied when the
# keyword is literally present (substring match), so the logic is transparent
# and easy to audit.  Factors > 1.0 increase a disease's share; the Stroke
# de-boost fires whenever NO stroke-specific indicator is present.
#
# Tuning guideline: keep boost factors between 1.2–2.0 (boost) or 0.4–0.7
# (de-boost).  Larger values risk over-correcting a genuinely ambiguous case.

_HEURISTIC_BOOSTS: dict[str, dict[str, float]] = {
    # ── Parkinson's ───────────────────────────────────────────────────────────
    "resting tremor":       {"Parkinson": 1.8},
    "pill-rolling":         {"Parkinson": 2.0},
    "bradykinesia":         {"Parkinson": 1.8},
    "freezing":             {"Parkinson": 1.5},
    "cogwheel":             {"Parkinson": 1.8},
    "tremor":               {"Parkinson": 1.4},
    "rigidity":             {"Parkinson": 1.4},
    "shuffling":            {"Parkinson": 1.4},
    "postural instability": {"Parkinson": 1.5},
    "rem sleep":            {"Parkinson": 1.6, "Dementia": 1.5},  # REM BD: hallmark of Lewy body dementia AND Parkinson's
    "hyposmia":             {"Parkinson": 1.4},
    "anosmia":              {"Parkinson": 1.2},
    "hypophonia":           {"Parkinson": 1.4},
    "masked face":          {"Parkinson": 1.5},
    "hypomimia":            {"Parkinson": 1.5},
    "falls":                {"Parkinson": 1.2},   # postural instability (mild boost)
    "sleep disturbance":    {"Parkinson": 1.2, "Dementia": 1.3},  # Lewy body dementia has more prominent sleep disturbance than Parkinson's alone

    # ── Alzheimer's / Dementia ────────────────────────────────────────────────
    "memory loss":          {"Alzheimer": 1.8, "Dementia": 1.5},
    "episodic memory":      {"Alzheimer": 1.6},
    "forgetfulness":        {"Alzheimer": 1.4, "Dementia": 1.3},
    "confusion":            {"Alzheimer": 1.3, "Dementia": 1.4},
    "disorientation":       {"Alzheimer": 1.3, "Dementia": 1.4},
    "aphasia":              {"Alzheimer": 1.4, "Dementia": 1.2},
    "cognitive decline":    {"Alzheimer": 1.4, "Dementia": 1.4},
    "word finding":         {"Alzheimer": 1.3, "Dementia": 1.2},

    # ── ALS ───────────────────────────────────────────────────────────────────
    "muscle atrophy":       {"ALS": 1.6},
    "muscle wasting":       {"ALS": 1.6},
    "fasciculation":        {"ALS": 1.9},
    "bulbar dysfunction":   {"ALS": 1.7},
    "bulbar palsy":         {"ALS": 1.7},
    "dysphagia":            {"ALS": 1.4},
    "limb weakness":        {"ALS": 1.5},
    "muscle weakness":      {"ALS": 1.3},
    "respiratory":          {"ALS": 1.3},
    "dysarthria":           {"ALS": 1.3, "Parkinson": 1.1, "Dementia": 1.2},  # prominent in frontotemporal dementia

    # ── Huntington's ─────────────────────────────────────────────────────────
    "chorea":               {"Huntington": 2.0},
    "involuntary movement": {"Huntington": 1.6},
    "choreiform":           {"Huntington": 2.0},
    "huntington":           {"Huntington": 2.0},

    # ── Stroke — positive boosts for genuinely stroke-specific features ───────
    "sudden":               {"Stroke": 1.5},
    "acute onset":          {"Stroke": 1.4},
    "hemiplegia":           {"Stroke": 1.6},
    "hemiparesis":          {"Stroke": 1.5},
    "hemorrhagic":          {"Stroke": 1.5},
    "ischaemic":            {"Stroke": 1.5},
    "ischemic":             {"Stroke": 1.5},
    "tia":                  {"Stroke": 1.6},
    "transient ischemic":   {"Stroke": 1.7},
    "infarction":           {"Stroke": 1.5},
    "atrial fibrillation":  {"Stroke": 1.4},
    "thrombosis":           {"Stroke": 1.4},
}

# When NONE of these keywords appear, Stroke is penalised (de-boosted).
# These are indicators that a symptom presentation is genuinely stroke-like.
_STROKE_SPECIFIC_INDICATORS = {
    "sudden", "acute onset", "hemiplegia", "hemiparesis", "hemorrhagic",
    "ischaemic", "ischemic", "tia", "transient ischemic", "infarction",
    "atrial fibrillation", "thrombosis", "embolism", "cerebrovascular",
    "brain attack", "one-sided", "one side", "hemiplegic",
}
_STROKE_DEBOOOST_FACTOR = 0.45   # reduce Stroke share by ~55% when no stroke indicator present


def _apply_heuristic_overlay(
    scores: dict,
    symptom_text: str,
    full_text: str = "",
) -> dict:
    """
    Adjust ensemble probabilities using disease-specific symptom keywords.

    Parameters
    ----------
    scores        : raw ensemble probabilities from ML models
    symptom_text  : template text WITHOUT family history / medications — used
                    for symptom-based boosts so family disease names don't leak
    full_text     : complete template text including family history — used only
                    for the soft family-history prior (small 1.2× boost)

    Applies multiplicative boosts for highly specific symptoms, applies a
    Stroke de-boost when no stroke-specific indicator is present, then applies
    a small contextual prior from family history, then re-normalises.
    """
    sym_lower  = symptom_text.lower()
    full_lower = (full_text or symptom_text).lower()
    adjusted   = dict(scores)

    # ── Positive symptom boosts (applied to symptom text only) ───────────────
    # NOTE: boost keys use short disease roots (e.g. "Parkinson") while the
    # actual label keys are full strings (e.g. "Parkinson's Disease").
    # Use substring matching so partial keys reliably find their target label.
    for keyword, boosts in _HEURISTIC_BOOSTS.items():
        if keyword in sym_lower:
            for disease_root, factor in boosts.items():
                root_lower = disease_root.lower()
                for full_label in list(adjusted.keys()):
                    if root_lower in full_label.lower():
                        adjusted[full_label] *= factor

    # ── Stroke de-boost (no stroke indicator in symptom text) ────────────────
    stroke_label = next((k for k in adjusted if "stroke" in k.lower()), None)
    if stroke_label:
        has_stroke_indicator = any(ind in sym_lower for ind in _STROKE_SPECIFIC_INDICATORS)
        if not has_stroke_indicator:
            adjusted[stroke_label] *= _STROKE_DEBOOOST_FACTOR

    # ── Soft family-history prior (uses full_text, applied last, modest boost)─
    # A parent with Parkinson's raises the prior for Parkinson's modestly — but
    # clinical presentation still dominates.  Cap at 1.2× so a single family
    # history mention cannot override a symptom-driven score.
    _FAMILY_HISTORY_BOOST_FACTOR = 1.2
    _FAMILY_DISEASE_KEYWORDS = {
        "parkinson":   "Parkinson",
        "alzheimer":   "Alzheimer",
        "dementia":    "Dementia",
        "als":         "ALS",
        "huntington":  "Huntington",
        "stroke":      "Stroke",
    }
    if "family history" in full_lower:
        # Extract text after "family history:" label for scoped matching
        fam_start = full_lower.find("family history")
        fam_text  = full_lower[fam_start: fam_start + 120]
        for kw, disease_root in _FAMILY_DISEASE_KEYWORDS.items():
            if kw in fam_text:
                root_lower = disease_root.lower()
                for full_label in list(adjusted.keys()):
                    if root_lower in full_label.lower():
                        adjusted[full_label] *= _FAMILY_HISTORY_BOOST_FACTOR
                        break   # one disease matched — don't double-apply

    # ── Re-normalise ─────────────────────────────────────────────────────────
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {d: v / total for d, v in adjusted.items()}

    return adjusted


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


def load_rf_model():
    """
    Load the separately saved Random Forest classifier.
    Returns the model or None if the file does not exist (e.g. before re-running
    phase3_ml.py with the updated code).
    """
    path = "random_forest_classifier.pkl"
    if not os.path.exists(path):
        print(
            "[INFO] random_forest_classifier.pkl not found — RF will be excluded "
            "from the ensemble. Re-run phase3_ml.py to generate it."
        )
        return None
    return joblib.load(path)


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
    """
    Load all scorer models. Returns a dict of loaded components.

    Ensemble members (used in score_template):
      clf        — best sklearn model (LinearSVC or LogisticRegression or RF)
      clf_rf     — Random Forest model (may be same as clf if RF was best)
      tok_bert / mod_bert — BioBERT (optional, None if not available)

    All three are averaged equally when present.  If RF is not yet trained
    (random_forest_classifier.pkl missing), the ensemble falls back to
    best-sklearn + BioBERT as before.
    """
    clf, vectorizer, le = load_sklearn_models()
    clf_rf              = load_rf_model()
    tok_bert, mod_bert  = load_biobert_model()
    return {
        "clf":        clf,
        "clf_rf":     clf_rf,
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


def _rf_probabilities(text: str, clf_rf, vectorizer) -> Optional[np.ndarray]:
    """
    Get probability distribution from the Random Forest classifier.
    RF natively supports predict_proba, so no softmax conversion is needed.
    Returns None if clf_rf is None (model not yet trained).
    """
    if clf_rf is None:
        return None
    try:
        vec = vectorizer.transform([text])
        return clf_rf.predict_proba(vec)[0]
    except Exception as e:
        print(f"[WARN] Random Forest inference failed: {e}")
        return None


def score_template(
    template_text: str,
    models: dict,
    symptom_text: str = "",
) -> dict:
    """
    Compute ensemble disease probability for a clinical template text string.

    Ensemble members (averaged equally when all present):
      1. Best sklearn model (LinearSVC / LogisticRegression / RandomForest)
      2. Random Forest (dedicated ensemble member — may overlap with #1 if
         RF happened to be the best model, in which case it gets 2× weight,
         which is intentional — a stronger model contributes more)
      3. BioBERT (optional — excluded if model files not present)

    Parameters
    ----------
    template_text : full text from ClinicalTemplate.to_text() (includes family history)
    models        : dict returned by load_all_models()
    symptom_text  : symptom-only text from ClinicalTemplate.to_symptom_text()
                    (excludes family history + medications to prevent disease
                    names in contextual fields from biasing the ML classifiers).
                    If omitted, falls back to template_text for backwards compat.

    Returns
    -------
    dict mapping disease label → probability (float, sums to ~1.0)
    """
    clf        = models["clf"]
    clf_rf     = models.get("clf_rf")       # None if not yet trained
    vectorizer = models["vectorizer"]
    le         = models["le"]
    tok_bert   = models["tok_bert"]
    mod_bert   = models["mod_bert"]

    # Use symptom-only text for ML classifiers so family history disease names
    # (e.g. "father had Parkinson's") do not contaminate the symptom score.
    ml_text = symptom_text if symptom_text.strip() else template_text

    # Collect all available probability vectors
    proba_list = []

    # Member 1 — best sklearn model (always present)
    proba_list.append(_sklearn_probabilities(ml_text, clf, vectorizer))

    # Member 2 — Random Forest (present after re-running phase3_ml.py)
    proba_rf = _rf_probabilities(ml_text, clf_rf, vectorizer)
    if proba_rf is not None:
        proba_list.append(proba_rf)

    # Member 3 — BioBERT (optional)
    if tok_bert is not None and mod_bert is not None:
        proba_bert = _biobert_probabilities(ml_text, tok_bert, mod_bert)
        if proba_bert is not None:
            proba_list.append(proba_bert)

    # Equal-weight average across all available members
    ensemble = np.mean(proba_list, axis=0)

    raw_scores = {
        disease: float(prob)
        for disease, prob in zip(le.classes_, ensemble)
    }

    # Apply heuristic calibration:
    #   symptom_text → symptom-specific boosts (no family history leak)
    #   template_text → soft family history prior (modest 1.2× only)
    return _apply_heuristic_overlay(
        raw_scores,
        symptom_text=ml_text,
        full_text=template_text,
    )


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
