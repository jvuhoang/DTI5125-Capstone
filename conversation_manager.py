"""
conversation_manager.py — Multi-Turn Routing + Session Management
=================================================================
Manages the full conversation lifecycle for the NORA chatbot:

  1. Initialise session state (history, template, scored flag)
  2. On each user turn:
       a. If factual question → route to RAG retriever
       b. Otherwise → extract slots into ClinicalTemplate
       c. Ask next follow-up question (ALL 6 fields collected first)
       d. Once all fields filled → trigger symptom scorer

Intent routing is heuristic-based (keyword + "?" detection).
The disease classifier is used to pre-filter RAG retrieval to
the most likely disease, improving retrieval precision.

All conversation history is passed to the LLM on every turn
so the model has full multi-turn context.
"""

import re
import random
from typing import Optional
import streamlit as st
import joblib

from template_filler import (
    ClinicalTemplate, extract_from_text, next_question, FIELD_PRIORITY
)
from rag_pipeline import PICOSRetriever, RAGAnswerGenerator
from symptom_scorer import score_template, format_text_scores


# ── Session initialisation ────────────────────────────────────────────────────

def init_session() -> None:
    """
    Initialise Streamlit session state for a new conversation.
    Safe to call on every page load — only sets defaults if not already present.
    """
    defaults = {
        "history":         [],
        "template":        ClinicalTemplate(),
        "scored":          False,
        "mode":            "intake",   # "intake" | "rag"
        "last_filled":     None,       # tracks which field was just filled
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    """Clear conversation state — used by the 'New conversation' button."""
    st.session_state.history      = []
    st.session_state.template     = ClinicalTemplate()
    st.session_state.scored       = False
    st.session_state.mode         = "intake"
    st.session_state.last_filled  = None


# ── Intent detection ──────────────────────────────────────────────────────────

# Clinical/research terms that signal a literature question.
# Question words (what, how, why…) are intentionally NOT included here —
# they are handled separately by _QUESTION_STARTER_WORDS.
# Combining question-word detection with medical keyword detection (below)
# prevents non-medical questions ("what is ice cream?") from matching.
_FACTUAL_KEYWORDS = {
    "treatment", "therapy", "drug", "medication", "study",
    "research", "trial", "evidence", "intervention", "outcome",
    "cause", "risk", "factor", "literature", "paper", "found",
    "published", "journal", "clinical", "prognosis", "survival",
    "prevalence", "incidence", "management", "diagnosis",
}

_SYMPTOM_KEYWORDS = {
    "tremor", "memory", "forget", "rigid", "stiff", "weak", "slow",
    "balance", "walk", "speech", "swallow", "fatigue", "pain",
    "confusion", "hallucin", "depress", "anxiety", "seizure",
    "paralys", "numb", "tingle", "chorea", "involuntary",
}


_QUESTION_STARTER_WORDS = {
    "what", "how", "which", "who", "when", "where", "why",
    "tell", "explain", "describe", "list", "compare",
}

# ── Medical relevance keyword set ────────────────────────────────────────────
# If NONE of these appear and the message isn't a question or symptom report,
# we treat it as off-topic small talk and respond with a polite redirect.
#
# Organised by category. Uses substring matching (kw in text_lower), so short
# roots like "tremor" match "tremors", "forget" matches "forgetting", etc.
# All entries are lowercase.

_MEDICAL_RELEVANCE_KEYWORDS = {

    # ── General medical / clinical vocabulary ─────────────────────────────────
    "symptom", "symptoms", "disease", "condition", "disorder", "syndrome",
    "treatment", "therapy", "drug", "medication", "medicine", "dose", "dosage",
    "diagnosis", "prognosis", "clinical", "patient", "doctor", "physician",
    "hospital", "clinic", "neurologist", "specialist", "medical", "health",
    "brain", "nerve", "neural", "cerebral", "neurological", "neurology",
    "neurodegenerative", "genetic", "hereditary", "inherited", "progressive",
    "chronic", "acute", "onset", "stage", "early-stage", "late-stage",

    # ── Research / literature vocabulary ──────────────────────────────────────
    "risk", "cause", "factor", "research", "study", "trial", "evidence",
    "intervention", "outcome", "prevalence", "incidence", "management",
    "literature", "paper", "journal", "published", "findings", "clinical trial",
    "randomised", "placebo", "cohort",

    # ── Alzheimer's Disease — names & synonyms ────────────────────────────────
    "alzheimer", "alzheimers", "alzheimer's",

    # ── Parkinson's Disease — names & synonyms ────────────────────────────────
    "parkinson", "parkinsons", "parkinson's", "parkinsonism",

    # ── ALS — names & synonyms ────────────────────────────────────────────────
    "als", "amyotrophic", "lateral sclerosis", "motor neuron", "lou gehrig", "mnd",

    # ── Huntington's Disease — names & synonyms ───────────────────────────────
    "huntington", "huntingtons", "huntington's",

    # ── Dementia / MCI — names & synonyms ────────────────────────────────────
    "dementia", "cognitive impairment", "mild cognitive", "mci",
    "vascular dementia", "lewy body", "frontotemporal", "forgetful",

    # ── Stroke — names & synonyms ─────────────────────────────────────────────
    "stroke", "cerebrovascular", "brain attack", "tia", "transient ischemic",
    "ischaemic", "ischemic", "haemorrhagic", "hemorrhagic", "thrombectomy",
    "thrombolysis", "alteplase", "anticoagulant", "antiplatelet",

    # ── Multiple sclerosis (mentioned in ontology / blocklist) ────────────────
    "multiple sclerosis", "sclerosis",

    # ── Motor symptoms ────────────────────────────────────────────────────────
    "tremor", "tremors", "shaking", "shaky", "shake", "trembling",
    "resting tremor", "pill-rolling",
    "bradykinesia", "slowness", "slow movement", "moving slowly",
    "akinesia", "freezing", "frozen movement",
    "hypokinesia", "reduced movement",
    "rigidity", "stiffness", "stiff", "muscle stiffness", "cogwheel",
    "dystonia", "spasm", "spasms", "muscle cramp",
    "chorea", "involuntary movement", "uncontrollable movement",
    "hypomimia", "masked face", "facial masking", "expressionless face",
    "kinesia paradoxica",
    "gait", "shuffling", "shuffle", "walking problem", "difficulty walking",
    "balance", "postural instability", "imbalance", "unsteady",
    "falls", "falling", "stumbling", "frequent falls",
    "paralysis", "paralysed", "paralyzed",

    # ── Weakness ──────────────────────────────────────────────────────────────
    "weakness", "weak", "muscle weakness", "limb weakness",
    "arm weakness", "leg weakness", "axial weakness", "trunk weakness",
    "muscle wasting", "atrophy", "muscle atrophy", "muscle loss",

    # ── Speech / voice / swallowing ───────────────────────────────────────────
    "speech", "slurred", "slurring", "dysarthria",
    "swallow", "swallowing", "dysphagia", "choking",
    "voice", "soft voice", "hypophonia", "quiet voice",
    "drooling", "drool", "sialorrhoea", "excessive saliva",
    "bulbar", "bulbar dysfunction", "bulbar palsy",
    "aphasia", "word finding", "language problem",

    # ── Cognitive / memory symptoms ───────────────────────────────────────────
    "memory", "forget", "forgetfulness", "forgetting", "memory loss",
    "cognitive", "cognitive decline", "brain fog", "confusion", "disorientation",
    "disoriented", "confused", "executive dysfunction", "reasoning problem",
    "concentration", "focus",

    # ── Neuropsychiatric symptoms ─────────────────────────────────────────────
    "hallucin", "seeing things", "hearing things", "visual hallucination",
    "depress", "depression", "low mood", "sadness", "mood",
    "anxiety", "anxious", "panic", "restless", "akathisia",
    "apathy", "personality change", "behavioural change", "behavior change",
    "pseudobulbar", "emotional lability", "involuntary crying",
    "mood swing", "irritable",

    # ── Autonomic / systemic symptoms ─────────────────────────────────────────
    "autonomic", "constipation", "bowel", "incontinence", "bladder control",
    "sweating", "hyperhidrosis",
    "smell", "anosmia", "hyposmia", "loss of smell",
    "breathing", "respiratory", "shortness of breath", "breathe",
    "sleep", "insomnia", "rem sleep", "sleep disorder", "sleep disturbance",
    "fatigue", "tired", "exhausted", "no energy",

    # ── Sensory symptoms ──────────────────────────────────────────────────────
    "numbness", "numb", "tingling", "paresthesia", "pins and needles",
    "burning sensation", "electric feeling",
    "vision", "blurry", "blurred", "double vision", "visual disturbance",
    "dizziness", "dizzy", "lightheaded", "vertigo",

    # ── Pain / other ──────────────────────────────────────────────────────────
    "pain", "neuropathic", "aching", "muscle pain", "headache", "migraine",
    "seizure", "seizures", "convulsion", "epilepsy", "fits",
    "fainting", "blackout", "blacking out", "loss of consciousness",
    "facial droop", "facial drooping", "facial weakness",
    "sudden numbness", "sudden weakness", "sudden confusion",
    "weight loss", "losing weight",
    "chest pain",

    # ── Medications & treatments (catch medication-name questions) ────────────
    "levodopa", "carbidopa", "donepezil", "memantine", "rivastigmine",
    "galantamine", "lecanemab", "donanemab", "aducanumab",
    "riluzole", "edaravone", "tetrabenazine", "deutetrabenazine",
    "antibiotic", "vaccine", "supplement", "vitamin", "supplement",
    "aspirin", "statin", "warfarin", "apixaban", "clopidogrel",
    "antidepressant", "antipsychotic", "sedative", "sleep aid",
}


def is_factual_query(text: str) -> bool:
    """
    Heuristic: classify user message as a factual/literature question
    (route to RAG) vs. a symptom description (route to template filler).

    A message is treated as a factual query when ALL THREE conditions hold:
      1. It is phrased as a question — ends with '?' OR starts with a
         question/instruction word (what, how, tell, explain …)
      2. It contains at least one clinical/research keyword (_FACTUAL_KEYWORDS)
         OR at least one recognised disease/symptom term (_MEDICAL_RELEVANCE_KEYWORDS)
      3. It has medical relevance — at least one term from _MEDICAL_RELEVANCE_KEYWORDS

    Requiring medical relevance (condition 3) prevents non-medical questions
    such as "what are the differences between ice cream and milk?" from
    accidentally triggering a literature retrieval.
    """
    text_lower  = text.lower().strip()
    first_word  = text_lower.split()[0] if text_lower.split() else ""
    has_q_mark  = "?" in text_lower
    has_factual = any(kw in text_lower for kw in _FACTUAL_KEYWORDS)
    has_medical = any(kw in text_lower for kw in _MEDICAL_RELEVANCE_KEYWORDS)
    starts_with_question_word = first_word in _QUESTION_STARTER_WORDS

    is_question = has_q_mark or starts_with_question_word
    return is_question and has_medical and (has_factual or has_medical)


# Phrases that strongly signal the user is describing their own symptoms
_SYMPTOM_REPORT_PHRASES = {
    "i have", "i've been", "i've had", "i am having", "i'm having",
    "i experience", "i feel", "i notice", "i've noticed",
    "i suffer", "i'm suffering", "patient has", "patient is",
    "he has", "she has", "they have", "he is", "she is",
    "started having", "been experiencing", "been feeling",
    "also have", "i also", "as well as",
}


def is_symptom_description(text: str) -> bool:
    """
    Returns True when a message looks like a personal symptom report
    rather than a literature question.  Used to break out of RAG mode.

    Requires BOTH a symptom-report phrase ("I have", "I've been", etc.)
    AND at least one symptom keyword — so plain statements like "I understand"
    don't trigger a false switch back to intake mode.
    """
    if is_factual_query(text):          # questions always go to RAG
        return False
    t = text.lower()
    has_report_phrase  = any(ph in t for ph in _SYMPTOM_REPORT_PHRASES)
    has_symptom_word   = any(kw in t for kw in _SYMPTOM_KEYWORDS)
    # Also check the synonym lookup — catches "difficulties sleeping" etc.
    try:
        from symptom_synonyms import lookup_symptoms
        has_synonym_hit = bool(lookup_symptoms(text))
    except Exception:
        has_synonym_hit = False
    return has_report_phrase and (has_symptom_word or has_synonym_hit)


def is_off_topic(text: str) -> bool:
    """
    Returns True when a message has no apparent medical relevance.

    The check is purely keyword-based — if none of the ~190 medical/disease/
    symptom terms in _MEDICAL_RELEVANCE_KEYWORDS appear in the text, the
    message is considered off-topic regardless of whether it is phrased as a
    question.  This prevents non-medical questions like
    "what are the differences between ice cream and milk?" from reaching the
    RAG pipeline, because they contain a question mark but zero medical content.
    """
    if not text.strip():
        return False
    text_lower = text.lower()
    # Any recognised medical keyword → medically relevant, not off-topic
    if any(kw in text_lower for kw in _MEDICAL_RELEVANCE_KEYWORDS):
        return False
    # No medical signal at all → off-topic
    return True


def predict_disease(text: str, clf, vectorizer, le) -> Optional[str]:
    """
    Use the TF-IDF classifier to predict the most likely disease for
    a query. Used to pre-filter RAG retrieval.
    Returns None if no clear prediction.
    """
    try:
        vec  = vectorizer.transform([text])
        pred = clf.predict(vec)[0]
        return le.inverse_transform([pred])[0]
    except Exception:
        return None


# Maps lowercase keywords → canonical DB disease_label values.
# Populated from Disease_entries_en.json synonyms + additional clinical terms.
# Longer/more specific phrases must come before short ones so they match first.
_DISEASE_KEYWORD_MAP = {
    # ── Alzheimer's Disease ───────────────────────────────────────────────────
    "alzheimer's disease":          "Alzheimer's Disease",
    "alzheimer disease":            "Alzheimer's Disease",
    "memory disease":               "Alzheimer's Disease",
    "alzheimer's":                  "Alzheimer's Disease",
    "alzheimer":                    "Alzheimer's Disease",

    # ── Parkinson's Disease ───────────────────────────────────────────────────
    "parkinson's disease":          "Parkinson's Disease",
    "parkinson disease":            "Parkinson's Disease",
    "parkinsonism":                 "Parkinson's Disease",
    "parkinson's":                  "Parkinson's Disease",
    "parkinson":                    "Parkinson's Disease",

    # ── ALS / Huntington's ────────────────────────────────────────────────────
    "amyotrophic lateral sclerosis":"ALS and Huntington's Disease",
    "motor neuron disease":         "ALS and Huntington's Disease",
    "lou gehrig's disease":         "ALS and Huntington's Disease",
    "lou gehrig":                   "ALS and Huntington's Disease",
    "als disease":                  "ALS and Huntington's Disease",
    "huntington's disease":         "ALS and Huntington's Disease",
    "huntington disease":           "ALS and Huntington's Disease",
    "huntington's":                 "ALS and Huntington's Disease",
    "huntington":                   "ALS and Huntington's Disease",
    "mnd":                          "ALS and Huntington's Disease",
    "als":                          "ALS and Huntington's Disease",

    # ── Dementia / MCI ────────────────────────────────────────────────────────
    "mild cognitive impairment":    "Dementia and Mild Cognitive Impairment",
    "cognitive impairment":         "Dementia and Mild Cognitive Impairment",
    "vascular dementia":            "Dementia and Mild Cognitive Impairment",
    "lewy body dementia":           "Dementia and Mild Cognitive Impairment",
    "frontotemporal dementia":      "Dementia and Mild Cognitive Impairment",
    "dementia":                     "Dementia and Mild Cognitive Impairment",

    # ── Stroke ────────────────────────────────────────────────────────────────
    "cerebrovascular accident":     "Stroke",
    "cerebrovascular":              "Stroke",
    "brain attack":                 "Stroke",
    "stroke":                       "Stroke",
}


def detect_mentioned_diseases(text: str) -> list:
    """
    Return the list of canonical disease labels explicitly named in the text.
    Preserves insertion order, deduplicates.
    """
    t = text.lower()
    seen   = set()
    result = []
    for kw, label in _DISEASE_KEYWORD_MAP.items():
        if kw in t and label not in seen:
            seen.add(label)
            result.append(label)
    return result


# ── Citation formatter ────────────────────────────────────────────────────────

def format_citations(picos_summary: list) -> str:
    """Format the PICOS summary list into a markdown citation block."""
    if not picos_summary:
        return ""
    lines = ["\n\n**Papers that informed this answer:**"]
    for p in picos_summary:
        title = p.get("title", "Unknown title")
        year  = p.get("year", "")
        pmid  = p.get("pmid", "")
        lines.append(f"- {title} ({year}) — PMID {pmid}")
    return "\n".join(lines)


# ── Conversational follow-up questions ───────────────────────────────────────
# Phrased naturally — each one feels like a follow-up in a real conversation,
# not a form field prompt.

FOLLOW_UP_QUESTIONS = {
    "primary_symptoms": (
        "What symptoms are you or the patient experiencing?"
    ),
    "duration": (
        "How long have these symptoms been going on?"
    ),
    "severity": (
        "How severe would you say the symptoms are? "
        "For example — mild and manageable, moderate and affecting some activities, "
        "or severe and significantly impacting daily life?"
    ),
    "age_gender": (
        "Could you tell me the patient's age and gender? "
        "This helps narrow down the most likely conditions."
    ),
    "family_history": (
        "Is there any family history of neurological conditions — "
        "things like dementia, Parkinson's, or stroke in a parent, "
        "sibling, or grandparent?"
    ),
    "current_medications": (
        "Is the patient currently taking any medications, or have they "
        "tried any treatments for these symptoms?"
    ),
}

# Brief acknowledgment lines — randomised to avoid robotic repetition
_ACKNOWLEDGMENTS = {
    "primary_symptoms": [
        "Thanks for describing that.",
        "Got it — I've noted those symptoms.",
        "Understood, that's helpful.",
    ],
    "duration": [
        "Got it, thanks.",
        "Understood.",
        "Thanks, that gives me a clearer picture.",
    ],
    "severity": [
        "Understood.",
        "Got it.",
        "Thanks for that.",
    ],
    "age_gender": [
        "Thanks.",
        "Got it.",
        "Noted.",
    ],
    "family_history": [
        "Thanks for sharing that.",
        "Appreciated — that background is useful.",
        "Noted.",
    ],
    "current_medications": [
        "Got it, thanks.",
        "Noted.",
        "Thanks for the detail.",
    ],
}


def _is_very_short_duration(duration_text: str) -> bool:
    """
    Returns True when the extracted duration is measured only in days —
    suggesting the symptoms are very recent (less than ~2 weeks old).
    Used to trigger a monitoring/consult advisory instead of a full assessment.

    Examples that return True:  "a few days", "3 days", "2 days", "days"
    Examples that return False: "a week", "2 weeks", "several months", "a year"
    """
    if not duration_text:
        return False
    d = duration_text.lower()
    has_days   = bool(re.search(r"\bday", d))
    has_longer = bool(re.search(r"\bweek|\bmonth|\byear|\bwhile|\blong", d))
    return has_days and not has_longer


# Shown when duration is days-only — advise monitoring rather than full assessment
_SHORT_DURATION_ADVISORY = (
    "Since these symptoms have only been present for **a few days**, it may be "
    "too early for a reliable clinical assessment.\n\n"
    "**If symptoms are mild**, we'd recommend monitoring them over the next "
    "1–2 weeks. Many short-lived neurological symptoms resolve on their own.\n\n"
    "**Please consult a doctor promptly if any of the following apply:**\n"
    "- Symptoms are worsening rapidly or are very severe\n"
    "- You notice sudden facial drooping, arm weakness, or slurred speech "
    "(these can be signs of stroke — call emergency services immediately)\n"
    "- There is difficulty breathing, swallowing, or walking\n"
    "- You or the patient has a known neurological condition or strong family history\n\n"
    "Would you like to continue the full assessment anyway, or come back "
    "if symptoms persist beyond a couple of weeks?"
)


def _get_newly_filled(before: ClinicalTemplate, after: ClinicalTemplate) -> list:
    """
    Return a list of field names that were unfilled before and filled after.
    A field is considered 'unfilled' if it was None OR the __YES__ placeholder.
    A field is considered 'filled' if it is now a real, non-placeholder value.
    This correctly detects the two-step family history flow:
      None → __YES__   (bare affirmative)
      __YES__ → real   (detail provided)
    """
    newly = []
    for field in FIELD_PRIORITY:
        before_val   = getattr(before, field)
        after_val    = getattr(after, field)
        was_unfilled = before_val is None or before_val == "__YES__"
        now_filled   = after_val is not None and after_val != "__YES__"
        if was_unfilled and now_filled:
            newly.append(field)
    return newly


def _acknowledge(field: str) -> str:
    """Return a short conversational acknowledgment for a just-filled field."""
    return random.choice(_ACKNOWLEDGMENTS.get(field, ["Got it."]))


def _build_score_intro(template: ClinicalTemplate) -> str:
    """
    Build a natural-language summary of what was collected,
    shown just before the disease probability panel.
    Includes family history and medications as clinical context notes.
    """
    parts = []
    if template.primary_symptoms:
        parts.append(f"symptoms of *{template.primary_symptoms}*")
    if template.duration:
        parts.append(f"present for *{template.duration}*")
    if template.severity:
        parts.append(f"described as *{template.severity}*")
    if template.age_gender:
        parts.append(f"patient is *{template.age_gender}*")

    summary = ", ".join(parts) if parts else "the details you've shared"

    intro = (
        f"Thanks — I now have a good picture of the situation. "
        f"Based on {summary}, here's what the clinical literature suggests "
        f"about the most likely conditions. Please keep in mind this is an "
        f"informational assessment only, not a medical diagnosis.\n\n"
    )

    # Add a clinical context note for family history and medications if present
    context_notes = []
    if template.family_history and template.family_history.lower() not in ("none", "no", "n/a"):
        context_notes.append(f"**Family history:** {template.family_history}")
    if template.current_medications and template.current_medications.lower() != "none":
        context_notes.append(f"**Current medications:** {template.current_medications}")

    if context_notes:
        intro += (
            "> 🩺 **Clinical context noted:**  \n"
            + "  \n".join(f"> {n}" for n in context_notes)
            + "\n> *These factors have been incorporated into the probability assessment.*\n\n"
        )

    intro += "👇 *See the probability panel below.*"
    return intro


# ── Main turn handler ─────────────────────────────────────────────────────────

def handle_turn(
    user_text:  str,
    retriever:  PICOSRetriever,
    rag:        RAGAnswerGenerator,
    clf,
    vectorizer,
    le,
) -> str:
    """
    Process one user turn. Updates session state and returns the assistant response.

    Special return value "__SCORE__" signals the Streamlit app to trigger
    st.rerun() and render the score panel below the chat.
    """
    st.session_state.history.append({"role": "user", "content": user_text})

    template = st.session_state.template

    # ── Off-topic guard ────────────────────────────────────────────────────────
    # Intercept messages with no medical relevance before they reach the RAG
    # pipeline.  IMPORTANT: suppress this guard when an intake is actively in
    # progress — short answers like "a few days", "moderate", or "65 male"
    # contain no medical keywords but are valid replies to a chatbot question.
    # We consider an intake active when the template has at least one field
    # filled and is not yet complete (i.e. there are still questions to ask).
    intake_active = (
        st.session_state.mode == "intake"
        and template.filled_count() > 0
        and not template.is_complete()
    )
    if is_off_topic(user_text) and not is_symptom_description(user_text) and not intake_active:
        off_topic_reply = (
            "I'm focused on neurological and neurodegenerative conditions. "
            "Feel free to describe symptoms you're concerned about, or ask me "
            "anything about Alzheimer's, Parkinson's, ALS, Huntington's, "
            "Dementia, or Stroke — I'm happy to help with those."
        )
        st.session_state.history.append({"role": "assistant", "content": off_topic_reply})
        return off_topic_reply

    # ── Mode auto-switch: RAG → intake ────────────────────────────────────────
    # If the user was in RAG mode (literature Q&A) but this message looks like
    # a symptom description, transparently switch back to clinical intake so
    # the template filler can collect symptoms normally.
    if st.session_state.mode == "rag" and is_symptom_description(user_text):
        st.session_state.mode = "intake"

    # ── Route 1: Factual/literature question → RAG ────────────────────────────
    if is_factual_query(user_text) or st.session_state.mode == "rag":
        st.session_state.mode = "rag"

        # Determine disease filter(s) for retrieval:
        # 1. If the question explicitly names diseases, use those (handles multi-disease)
        # 2. If the question is generic, predict from the patient's recorded symptoms
        # 3. Never bleed a previous session's symptoms into an unrelated question
        mentioned = detect_mentioned_diseases(user_text)
        if len(mentioned) > 1:
            # Multi-disease question — retrieve per disease and merge
            filter_diseases = mentioned
        elif len(mentioned) == 1:
            filter_diseases = mentioned          # single explicit disease
        elif template.primary_symptoms:
            # Generic question — predict from what the patient already described
            predicted = predict_disease(template.primary_symptoms, clf, vectorizer, le)
            filter_diseases = [predicted] if predicted else []
        else:
            filter_diseases = []                 # no filter — search everything

        result = rag.answer(
            user_text, k=5, filter_diseases=filter_diseases
        )
        answer = result["answer"]
        if "Papers that informed" not in answer and "PMID" not in answer:
            answer += format_citations(result.get("picos_summary", []))
        st.session_state.history.append({"role": "assistant", "content": answer})
        return answer

    # ── Route 2: Slot filling ─────────────────────────────────────────────────
    # Snapshot template BEFORE extraction so we can detect what changed.
    before_snapshot = ClinicalTemplate(
        age_gender          = template.age_gender,
        primary_symptoms    = template.primary_symptoms,
        duration            = template.duration,
        severity            = template.severity,
        family_history      = template.family_history,
        current_medications = template.current_medications,
    )

    template = extract_from_text(user_text, template)
    st.session_state.template = template

    newly_filled = _get_newly_filled(before_snapshot, template)
    ack          = _acknowledge(newly_filled[0]) if newly_filled else ""

    # ── Short-duration advisory ───────────────────────────────────────────────
    # If the user just provided a duration measured only in days, the symptoms
    # are very recent. Rather than continuing straight to a disease assessment,
    # advise them to monitor and consult a doctor, and offer to continue anyway.
    if "duration" in newly_filled and _is_very_short_duration(template.duration):
        st.session_state.history.append(
            {"role": "assistant", "content": _SHORT_DURATION_ADVISORY}
        )
        return _SHORT_DURATION_ADVISORY

    # ── Route 3: Still questions to ask → ask next one ───────────────────────
    # All 6 fields are collected before triggering the scorer.
    nq = next_question(template)
    if nq:
        if not newly_filled and user_text.strip():
            # User said something but nothing was extracted — likely a misspelling
            # or an unclear answer. Ask again with a polite clarification prefix.
            _RETRY_PREFIXES = [
                "Sorry, I didn't quite catch that — could you rephrase? ",
                "I'm not sure I understood that — could you clarify? ",
                "Apologies, I couldn't parse that response — could you try again? ",
            ]
            import random as _random
            prefix   = _random.choice(_RETRY_PREFIXES)
            response = prefix + nq
        else:
            response = f"{ack} {nq}".strip() if ack else nq
        st.session_state.history.append({"role": "assistant", "content": response})
        return response

    # ── Route 4: All questions answered → trigger score panel ────────────────
    if not st.session_state.scored:
        st.session_state.scored = True
        intro   = _build_score_intro(template)
        # Compute scores inline so we can show █░ bars in the chat bubble
        _models = {"clf": clf, "vectorizer": vectorizer, "le": le,
                   "tok_bert": None, "mod_bert": None}
        try:
            _scores  = score_template(template.to_text(), _models)
            bars_msg = "\n\n" + format_text_scores(_scores, template)
        except Exception:
            bars_msg = ""
        st.session_state.history.append(
            {"role": "assistant", "content": intro + bars_msg}
        )
        return "__SCORE__"

    # ── Route 5: Already scored — check if new symptoms were added this turn ──
    # Compare against before_snapshot (captured at the start of this turn),
    # NOT against template (which Route 2 already updated). This is the correct
    # diff — Route 2 has already written any new symptoms into template.
    symptoms_before = before_snapshot.primary_symptoms or ""
    symptoms_after  = template.primary_symptoms or ""
    new_symptom_found = symptoms_after != symptoms_before

    if new_symptom_found:
        # What was just added?
        newly_added = symptoms_after.replace(symptoms_before, "").strip(", ")
        ack = (
            f"Got it — I've added *{newly_added}* to the picture. "
            f"Updating the assessment now.\n\n"
        )
        intro   = ack + _build_score_intro(template)
        _models = {"clf": clf, "vectorizer": vectorizer, "le": le,
                   "tok_bert": None, "mod_bert": None}
        try:
            _scores  = score_template(template.to_text(), _models)
            bars_msg = "\n\n" + format_text_scores(_scores, template)
        except Exception:
            bars_msg = ""
        st.session_state.history.append(
            {"role": "assistant", "content": intro + bars_msg}
        )
        return "__SCORE__"

    # No new symptoms — invite literature questions
    closing = (
        "I've already run the assessment above. Feel free to ask me anything "
        "about the research literature — for example:\n"
        "- *What treatments have been studied for these symptoms?*\n"
        "- *What does the evidence say about disease progression?*\n"
        "- *Which clinical trials are relevant to this condition?*"
    )
    st.session_state.history.append({"role": "assistant", "content": closing})
    st.session_state.mode = "rag"
    return closing


# ── Greeting ──────────────────────────────────────────────────────────────────

GREETING = (
    "Hi there! I'm **NORA**, a clinical literature assistant for "
    "neurodegenerative and neurological diseases.\n\n"
    "I can help you in two ways:\n"
    "- **Symptom assessment** — describe what the patient is experiencing "
    "and I'll guide you through a short intake, then show disease probability "
    "scores based on the clinical literature.\n"
    "- **Literature questions** — ask me anything about treatments, outcomes, "
    "or research findings for Alzheimer's, Parkinson's, ALS, Huntington's, "
    "Dementia, or Stroke.\n\n"
    "To get started — what symptoms are you or the patient experiencing?"
)


def get_greeting() -> str:
    return GREETING
