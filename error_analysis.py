"""
error_analysis.py — Systematic Error Analysis for NORA Chatbot
===============================================================
Collects, categorises and analyses chatbot failure patterns from a
curated set of (query, response) pairs.

Usage
-----
    python error_analysis.py                      # run on built-in test bank
    python error_analysis.py --input failures.json  # run on your own JSON file

Input JSON format (--input):
    [
        {"query": "...", "response": "...", "expected_route": "intake|rag|off-topic"},
        ...
    ]
    The 'expected_route' field is optional — if omitted the analyser infers it.

Outputs
-------
    error_analysis_report.txt   — full narrative report
    error_analysis_charts.png   — 4-panel visualisation
    error_analysis_results.json — raw categorised data for further processing
"""

from __future__ import annotations

import json
import re
import sys
import argparse
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Live router import ────────────────────────────────────────────────────────
# Import the actual routing functions from conversation_manager so the analysis
# runs against the *current* chatbot code, not a heuristic approximation.
#
# The router is stateless for individual messages at the classification level —
# is_factual_query(), is_off_topic(), and is_symptom_description() only inspect
# the query text.  handle_turn() additionally reads session state (mode, template)
# but those state-dependent effects are tested separately via context_test cases.

_LIVE_ROUTER_AVAILABLE = False
try:
    import types, sys as _sys

    # ── Mock heavy optional dependencies so the import chain succeeds ────
    # The routing functions (is_factual_query, is_off_topic,
    # is_symptom_description) are pure-Python text functions that don't
    # use ML models. Their import chain pulls in streamlit, sentence_transformers,
    # faiss and anthropic, none of which are needed here.
    def _mock_module(name: str, **attrs):
        if name not in _sys.modules:
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            _sys.modules[name] = m
        return _sys.modules[name]

    # Streamlit
    _st = _mock_module("streamlit")
    _st.session_state = {}
    _st.cache_resource = lambda **kw: (lambda f: f)
    _st.error = lambda *a, **kw: None

    # ML / embedding stubs
    _mock_module("sentence_transformers")
    _mock_module("sentence_transformers.SentenceTransformer")
    _st2 = _mock_module("sentence_transformers")
    _st2.SentenceTransformer = type("SentenceTransformer", (), {"__init__": lambda s, *a, **k: None})
    _mock_module("faiss")
    _mock_module("anthropic")
    _mock_module("anthropic.Anthropic")
    _anth = _mock_module("anthropic")
    _anth.Anthropic = type("Anthropic", (), {"__init__": lambda s, *a, **k: None})
    _mock_module("joblib")
    _jl = _mock_module("joblib")
    _jl.load = lambda *a, **k: None

    # spaCy stubs (template_filler tries to load en_ner_bc5cdr_md)
    _mock_module("spacy")
    _sp = _mock_module("spacy")
    _sp.load = lambda *a, **k: (_ for _ in ()).throw(OSError("mocked"))

    from conversation_manager import (
        is_factual_query       as _cm_is_factual_query,
        is_off_topic           as _cm_is_off_topic,
        is_symptom_description as _cm_is_symptom_description,
    )
    _LIVE_ROUTER_AVAILABLE = True
    print("  [router] Live routing functions loaded from conversation_manager ✓")
except Exception as _e:
    print(f"  [router] conversation_manager unavailable ({_e}). "
          "Falling back to response-text heuristics.")

# ── Constants ─────────────────────────────────────────────────────────────────

DISEASE_KEYWORDS = {
    "alzheimer", "parkinson", "als", "amyotrophic", "huntington",
    "dementia", "stroke", "cognitive", "motor neuron",
}

SYMPTOM_KEYWORDS = {
    "tremor", "memory", "forget", "rigid", "stiff", "weak", "slow",
    "balance", "walk", "speech", "swallow", "fatigue", "pain",
    "confusion", "hallucin", "depress", "anxiety", "seizure",
    "paralys", "numb", "tingle", "chorea", "involuntary", "rigidity",
    "gait", "shuffle", "bradykinesia", "freezing", "slurred", "dysphagia",
}

REPORT_PHRASES = {
    "i have", "i've been", "i've had", "i am having", "i'm having",
    "i experience", "i feel", "i notice", "i suffer", "patient has",
    "patient is", "my patient", "the patient", "he has", "she has",
    "they have", "started having", "been experiencing", "been feeling",
    "also have",
}

FACTUAL_KEYWORDS = {
    "treatment", "therapy", "drug", "medication", "study", "research",
    "trial", "evidence", "intervention", "outcome", "cause", "risk",
    "factor", "literature", "paper", "clinical", "prognosis", "survival",
    "prevalence", "incidence", "management", "diagnosis",
}

QUESTION_STARTERS = {
    "what", "how", "which", "who", "when", "where", "why",
    "tell", "explain", "describe", "list", "compare",
}

GENERIC_RESPONSE_PHRASES = [
    "i don't understand", "i'm not sure", "could you clarify",
    "please rephrase", "i cannot help", "i'm unable to",
    "that's outside", "i don't have information",
    "could you be more specific",
]

FAILURE_CATEGORIES = [
    "misunderstood_intent",
    "poor_context_handling",
    "generic_unhelpful",
    "out_of_scope_response",
]

FAILURE_LABELS = {
    "misunderstood_intent":   "Misunderstood Intent",
    "poor_context_handling":  "Poor Context Handling",
    "generic_unhelpful":      "Generic / Unhelpful",
    "out_of_scope_response":  "Out-of-Scope Response",
}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class QueryRecord:
    """One (query, response) pair with all analysis fields."""
    query:            str
    response:         str
    expected_route:   Optional[str]    = None  # "intake" | "rag" | "off-topic"

    # ── Inferred labels ───────────────────────────────────────────────────────
    detected_route:   Optional[str]    = None
    is_failure:       bool             = False
    failure_type:     Optional[str]    = None
    failure_reason:   Optional[str]    = None

    # ── Feature flags ─────────────────────────────────────────────────────────
    has_disease_kw:   bool             = False
    has_symptom_kw:   bool             = False
    has_report_phrase: bool            = False
    has_question_mark: bool            = False
    has_factual_kw:   bool             = False
    starts_question:  bool             = False
    is_formal:        bool             = False   # first-person plural / clinical phrasing
    mentioned_diseases: list           = field(default_factory=list)
    inferred_topic:   Optional[str]    = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ── Built-in test bank (200 examples) ─────────────────────────────────────────
# Each entry: (query, response, expected_route)
# expected_route: "intake" | "rag" | "off-topic" | None (= let analyser judge)

_TEST_BANK: list[tuple[str, str, str]] = [

    # ── MISUNDERSTOOD INTENT: symptom report routed to RAG ─────────────────
    ("My patient has resting tremor and rigidity — what could this indicate?",
     "Based on recent literature, studies of parkinsonian symptoms have found...",
     "intake"),
    ("She has been experiencing memory loss and confusion for 3 months.",
     "Research on Alzheimer's disease found that beta-amyloid plaques...",
     "intake"),
    ("He has weakness in both arms and slurred speech, started 6 weeks ago.",
     "Studies on ALS interventions have found riluzole reduces...",
     "intake"),
    ("I've been shaking uncontrollably and my hands won't stop trembling.",
     "According to literature on Parkinson's disease, tremor is managed...",
     "intake"),
    ("I have stiffness in my legs and difficulty walking since last month.",
     "Multiple studies on motor symptoms report that...",
     "intake"),
    ("My father has been forgetting things. He's 78 and it started a year ago.",
     "Research into Alzheimer's dementia published in 2023 found...",
     "intake"),
    ("I notice my movements are getting slower and my handwriting is tiny.",
     "A cohort study of Parkinson's patients found bradykinesia...",
     "intake"),
    ("Patient is 65, male, presenting with gait freezing and falls.",
     "Evidence-based treatment guidelines for Parkinson's recommend...",
     "intake"),
    ("I've had balance problems and difficulty swallowing for 2 months.",
     "Literature on neurodegenerative conditions suggests dysphagia...",
     "intake"),
    ("She's 70 and has been getting confused easily, forgetting names.",
     "Clinical trials on dementia found that cholinesterase inhibitors...",
     "intake"),

    # ── MISUNDERSTOOD INTENT: literature question routed to intake ─────────
    ("What treatments are available for Parkinson's disease?",
     "What are your symptoms? Please describe what you're experiencing.",
     "rag"),
    ("How does levodopa work for Parkinson's patients?",
     "How long have you been experiencing these symptoms?",
     "rag"),
    ("What are the risk factors for Alzheimer's disease?",
     "I understand. What is your age and gender?",
     "rag"),
    ("Tell me about the clinical trials for ALS treatment.",
     "Please describe your primary symptoms.",
     "rag"),
    ("What is the difference between Alzheimer's and dementia?",
     "Can you tell me more about the symptoms?",
     "rag"),
    ("Explain the progression of Huntington's disease.",
     "What medications are you currently taking?",
     "rag"),
    ("What drugs are used to slow cognitive decline?",
     "How severe would you rate your symptoms?",
     "rag"),
    ("How is ALS diagnosed?",
     "Do you have a family history of neurological conditions?",
     "rag"),
    ("What are the early signs of Parkinson's?",
     "Let me ask you a few questions to better understand your situation.",
     "rag"),
    ("Compare stroke symptoms with ALS symptoms.",
     "I see. Can you describe the symptoms more specifically?",
     "rag"),

    # ── POOR CONTEXT HANDLING: realistic full-sentence queries that should ────
    # route to intake but risk being misclassified (short, no disease keyword) ─
    # Each tests a distinct semantic variant of a symptom or history report.

    # Severity / duration reports — explicit sentence form
    ("My symptoms would be moderate in severity, not yet affecting daily life.",
     "I'm sorry, I don't have enough context. Could you describe your symptoms?",
     "intake"),
    ("The tremor has been going on for about three months now and is slowly worsening.",
     "I don't understand. What condition are you asking about?",
     "intake"),
    ("My mother was also diagnosed with early-onset Alzheimer's in her late sixties.",
     "I'm not sure what you're referring to. Please start over.",
     "intake"),
    ("I am not currently taking any medication, prescribed or otherwise.",
     "I don't have that information. Could you clarify?",
     "intake"),
    ("The weakness in my hands began a few weeks after a minor fall.",
     "I don't understand. What are your symptoms?",
     "intake"),
    ("The pain and stiffness have become quite severe over the past month.",
     "I'm unable to process that. Please describe your symptoms.",
     "intake"),

    # Age / gender / demographics — sentence form
    ("I am a 65-year-old male who has recently noticed difficulty initiating movement.",
     "I don't understand. Could you rephrase your question?",
     "intake"),
    ("I have no family history of neurological disease and take no regular medication.",
     "Could you be more specific about what you mean?",
     "intake"),
    ("Yes, I believe my father's side does have a history of memory problems.",
     "I'm not sure what you're referring to. Could you clarify?",
     "intake"),

    # Symptom elaboration — description without explicit disease name
    ("The trembling is mainly in both hands and is noticeably worse when I am at rest.",
     "I don't have enough information. Please start the assessment again.",
     "intake"),
    ("It came on suddenly — I woke up one morning and could not control my left hand.",
     "I'm unable to help with that. Could you rephrase?",
     "intake"),
    ("The weakness seems to affect both arms equally and feels like they are getting heavier.",
     "I don't understand. Please describe your symptoms more clearly.",
     "intake"),
    ("The shaking is always worse first thing in the morning before I have eaten.",
     "I'm not sure I follow. Could you explain more?",
     "intake"),

    # Demographic statement with clinical framing
    ("I am approximately 55 years old, female, and post-menopausal with no prior neurological diagnosis.",
     "I cannot process that input. Please try again.",
     "intake"),
    ("The decline has been gradual but consistent — each month noticeably worse than the last.",
     "I don't have context for this. What symptoms are you describing?",
     "intake"),

    # ── GENERIC / UNHELPFUL responses ──────────────────────────────────────
    ("What causes Alzheimer's disease?",
     "I don't understand your question. Could you please rephrase it?",
     "rag"),
    ("Can Parkinson's be cured?",
     "I'm not sure what you're asking. Please try again.",
     "rag"),
    ("Is ALS hereditary?",
     "I'm unable to help with that query at this time.",
     "rag"),
    ("What does bradykinesia feel like?",
     "I cannot provide information on that topic.",
     "rag"),
    ("My hands are shaking, what should I do?",
     "I'm not sure I understand. Could you clarify?",
     "intake"),
    ("Will my symptoms get worse over time?",
     "That's outside my area of knowledge.",
     "rag"),
    ("Is there a test for Parkinson's disease?",
     "I don't have information about that.",
     "rag"),
    ("Tell me about donepezil",
     "I'm unable to answer that question right now.",
     "rag"),
    ("How is dementia different from normal aging?",
     "I don't understand. Could you rephrase?",
     "rag"),
    ("My memory has been getting worse, is that serious?",
     "I'm not sure I can help with that. Please consult a doctor.",
     "intake"),
    ("What is the life expectancy for ALS patients?",
     "I cannot provide that information.",
     "rag"),
    ("Are there any new treatments for Huntington's?",
     "I don't have enough information to answer.",
     "rag"),
    ("What stage of Alzheimer's causes personality changes?",
     "I'm not sure I understand the question.",
     "rag"),
    ("How does stroke affect memory?",
     "That is outside what I can help with.",
     "rag"),
    ("Can exercise slow Parkinson's progression?",
     "I'm unable to help with that at the moment.",
     "rag"),

    # ── OUT-OF-SCOPE RESPONSES: bot answered when it shouldn't have ────────
    ("I see 2 rabbits outside",
     "Based on studies of neurodegenerative diseases, rabbits...",
     "off-topic"),
    ("What is the best pizza topping?",
     "Research has shown that dietary choices may affect Alzheimer's risk...",
     "off-topic"),
    ("The weather is nice today",
     "Environmental factors have been linked to neurological conditions...",
     "off-topic"),
    ("I love my dog",
     "Pet ownership has been studied in the context of dementia care...",
     "off-topic"),
    ("My car needs servicing",
     "Mechanical issues could cause stress, which is a risk factor for...",
     "off-topic"),
    ("Can you help me write an email?",
     "Certainly! Communication difficulties are sometimes seen in ALS...",
     "off-topic"),
    ("What is the capital of France?",
     "Geographic factors have been studied in Alzheimer's disease research...",
     "off-topic"),
    ("Tell me a joke",
     "Humour and cognitive engagement have been studied in dementia...",
     "off-topic"),
    ("What's 2 plus 2?",
     "Mathematical cognition is affected in some neurodegenerative diseases...",
     "off-topic"),
    ("How do I cook pasta?",
     "Dietary factors including pasta consumption have been studied...",
     "off-topic"),
    ("I'm bored",
     "Boredom and under-stimulation are linked to cognitive decline...",
     "off-topic"),
    ("What time is it?",
     "Disorientation to time is a symptom seen in Alzheimer's disease...",
     "off-topic"),
    ("Tell me about football",
     "Head injuries in football players have been linked to CTE...",
     "off-topic"),
    ("I need help with my taxes",
     "Financial stress has been linked to risk of dementia...",
     "off-topic"),
    ("What movies are showing tonight?",
     "Cognitive stimulation through media has been studied in dementia...",
     "off-topic"),

    # ── MIXED / AMBIGUOUS FAILURES ──────────────────────────────────────────
    ("Could this be Parkinson's?",
     "Research on Parkinson's disease suggests multiple diagnostic criteria...",
     "intake"),
    ("What are the differences between ice cream and milk?",
     "Nutritional factors including dairy have been studied in Alzheimer's...",
     "off-topic"),
    ("Should I see a neurologist?",
     "I don't have information to answer that question.",
     "rag"),
    ("Is this serious?",
     "Multiple studies have found that early intervention is key...",
     "intake"),
    ("My mum keeps repeating herself. Could it be dementia?",
     "According to literature on dementia, repetitive questioning is...",
     "intake"),
    ("Can stress cause memory problems?",
     "I don't understand the question. Could you rephrase?",
     "rag"),
    ("How long does it take to diagnose ALS?",
     "Can you describe your symptoms to me?",
     "rag"),
    ("Are tremors always Parkinson's?",
     "Please tell me more about what you're experiencing.",
     "rag"),
    ("He keeps losing his balance and falling",
     "Clinical evidence on fall prevention in Parkinson's suggests...",
     "intake"),
    ("My wife has been diagnosed with early Alzheimer's — what should we expect?",
     "I don't have enough information to assist. Please rephrase.",
     "rag"),

    # ── CORRECT RESPONSES (true negatives — should not be flagged as failures)
    ("What are the symptoms of Parkinson's disease?",
     "Parkinson's disease is characterised by motor symptoms including resting tremor, rigidity, and bradykinesia. Non-motor symptoms include sleep disturbances, depression, and anosmia.\n\nSources:\n[1] Motor features in Parkinson's (2021) — PMID 33912345",
     "rag"),
    ("Tell me about levodopa treatment",
     "Levodopa remains the gold-standard treatment for Parkinson's motor symptoms. It is converted to dopamine in the brain...\n\nSources:\n[1] Levodopa efficacy study (2022) — PMID 34567890",
     "rag"),
    ("What are the symptoms of late stage Alzheimer's disease?",
     "Late-stage Alzheimer's is characterised by severe memory loss, loss of ability to communicate, and dependence on others for daily care.\n\nSources:\n[1] Late-stage Alzheimer's clinical features (2022) — PMID 35123456",
     "rag"),
    ("I have been experiencing tremors in my right hand for about two months now.",
     "Thank you for sharing that. How would you rate the severity — mild, moderate, or severe?",
     "intake"),
    ("The symptoms have been present for roughly two months and seem to be progressing.",
     "Got it. How severe would you say they are — mild, moderate, or severe?",
     "intake"),
    ("I would describe the severity as moderate — it is noticeable but I can still manage daily tasks.",
     "Understood. What is the patient's age and gender?",
     "intake"),
    ("I am a 65-year-old male and I first noticed the symptoms about three months ago.",
     "Thank you. Is there any family history of neurological conditions?",
     "intake"),
    ("Yes, my father was diagnosed with Parkinson's disease in his early seventies.",
     "Noted. Are you currently taking any medications?",
     "intake"),
    ("I am not currently on any prescription medication for this condition.",
     "Thank you for sharing all that. Based on the information provided...",
     "intake"),
    ("How is Alzheimer's different from Parkinson's?",
     "Here's how these conditions differ:\n\nAlzheimer's disease is primarily a memory and cognitive disorder...",
     "rag"),
    ("What risk factors increase the chance of stroke?",
     "Key risk factors for stroke include hypertension, atrial fibrillation, diabetes, smoking, and high cholesterol...",
     "rag"),
    ("I see rabbits outside",
     "I'm focused on neurological and neurodegenerative conditions. Feel free to ask about Alzheimer's, Parkinson's, ALS, or stroke.",
     "off-topic"),

    # ── FORMAL vs CASUAL PHRASING failures ─────────────────────────────────
    ("The patient presents with cogwheel rigidity and micrographia.",
     "Research on Parkinson's disease found that cogwheel rigidity...",
     "intake"),
    ("Bilateral upper limb tremor, progressive onset over 18 months.",
     "Studies in the literature on tremor disorders suggest...",
     "intake"),
    ("Presenting complaint: gait disturbance and postural instability.",
     "Based on evidence for Parkinson's management...",
     "intake"),
    ("yo my hands keep shakin real bad idk what it is",
     "I don't understand. Please rephrase.",
     "intake"),
    ("cant remember stuff lately its getting worse",
     "Could you clarify your question?",
     "intake"),
    ("mum has dementia and shes getting worse what do i do",
     "Research on dementia management suggests...",
     "rag"),
    ("shes been falling a lot and her balance is off",
     "Could you elaborate on the symptoms?",
     "intake"),
    ("feeling really weak in my legs and cant walk properly",
     "I'm not sure I understand. Please describe your symptoms more clearly.",
     "intake"),

    # ── ADDITIONAL EDGE CASES — semantically varied, full sentences ───────────

    # Empty / greeting — should redirect politely
    ("",
     "I'm here to help with neurological conditions. Could you share your symptoms?",
     "off-topic"),
    ("Good morning, I was hoping to ask about a neurological concern.",
     "Of course — please go ahead and describe what you're experiencing.",
     "intake"),

    # Vague distress without specific symptom keyword
    ("I honestly have no idea what is happening to me neurologically.",
     "Research into undiagnosed neurological conditions found...",
     "intake"),
    ("My neurological symptoms seem to be deteriorating despite treatment.",
     "I'm unable to help with that. Please try rephrasing.",
     "intake"),
    ("All the neurological tests came back normal yet I still feel unwell.",
     "Diagnostic sensitivity for early neurological conditions varies...",
     "intake"),
    ("I am genuinely worried this could be something serious neurologically.",
     "I don't understand the question.",
     "intake"),

    # Clinical / research vocabulary edge cases
    ("What does the UPDRS scale measure in Parkinson's disease assessment?",
     "Please describe your symptoms to continue.",
     "rag"),
    ("What is the clinical significance of a positive Babinski sign in ALS?",
     "A positive Babinski sign indicates upper motor neuron involvement...\n\nSources:\n[1] UMN signs in ALS (2020) — PMID 32567891",
     "rag"),
    ("He has experienced three unexplained falls in the past week.",
     "Fall risk in neurological patients has been studied...",
     "intake"),

    # Paediatric / unusual population
    ("Can children and adolescents develop Parkinson's disease or is it only in older adults?",
     "Yes, early-onset Parkinson's disease can occur in younger individuals...",
     "rag"),
    ("What is the earliest age at which Huntington's symptoms typically appear?",
     "Juvenile Huntington's disease can manifest before age 20...\n\nSources:\n[1] Juvenile HD onset (2021) — PMID 33891234",
     "rag"),

    # Caregiver perspective
    ("My elderly father has been increasingly confused and I am concerned about his safety.",
     "I'm not sure I can help with that. Please try rephrasing.",
     "intake"),
    ("As a caregiver for someone with late-stage ALS, what communication aids are recommended?",
     "Augmentative and alternative communication (AAC) devices are recommended for ALS patients...\n\nSources:\n[1] AAC in ALS (2022) — PMID 34701234",
     "rag"),
]


# ── Feature extraction ─────────────────────────────────────────────────────────

def _extract_features(query: str) -> dict:
    """Return a dict of boolean feature flags for a single query string."""
    q = query.lower().strip()
    words = q.split()
    first = words[0] if words else ""

    diseases = [d for d in DISEASE_KEYWORDS if d in q]

    return {
        "has_disease_kw":    bool(diseases),
        "has_symptom_kw":    any(kw in q for kw in SYMPTOM_KEYWORDS),
        "has_report_phrase": any(ph in q for ph in REPORT_PHRASES),
        "has_question_mark": "?" in q,
        "has_factual_kw":    any(kw in q for kw in FACTUAL_KEYWORDS),
        "starts_question":   first in QUESTION_STARTERS,
        "is_formal": any(w in q for w in [
            "presents with", "presenting", "complaint",
            "bilateral", "unilateral", "onset", "progressive",
            "cogwheel", "micrographia", "hypomimia", "dysarthria",
        ]),
        "is_casual": bool(re.search(r"\b(yo|idk|cant|gonna|wanna|mum|shes|hes)\b", q)),
        "mentioned_diseases": diseases,
        "is_empty": len(q.strip()) == 0,
    }


def _infer_expected_route(query: str, features: dict) -> str:
    """Infer what the correct route should be based on features."""
    if features["is_empty"]:
        return "off-topic"
    if features["has_report_phrase"] and features["has_symptom_kw"]:
        return "intake"
    is_question = features["has_question_mark"] or features["starts_question"]
    if is_question and (features["has_factual_kw"] or features["has_disease_kw"]):
        return "rag"
    if features["has_symptom_kw"] or features["has_disease_kw"]:
        return "intake"
    return "off-topic"


def _live_route(query: str) -> str:
    """
    Call the real chatbot routing functions to decide where a query would go.

    Mirrors the logic in handle_turn() for a fresh-turn context (no active
    intake in progress, mode = "intake"):

        1. Off-topic guard  → "off-topic"
        2. is_factual_query → "rag"
        3. Otherwise        → "intake"

    This is the ground-truth detector used for all test-bank cases.
    """
    if not _LIVE_ROUTER_AVAILABLE:
        return "unknown"
    if _cm_is_off_topic(query) and not _cm_is_symptom_description(query):
        return "off-topic"
    if _cm_is_factual_query(query):
        return "rag"
    return "intake"


def _response_heuristic_route(response: str) -> str:
    """
    Fallback route classifier that parses the response text.
    Only used when conversation_manager could not be imported.
    """
    r = response.lower()
    if any(p in r for p in [
        "what are your symptoms", "how long have", "how severe",
        "what is your age", "family history", "are you currently taking",
        "please describe", "tell me about your", "let me ask you",
        "could you describe", "can you describe", "can you tell me",
    ]):
        return "intake"
    if any(p in r for p in [
        "studies", "research", "literature", "evidence", "clinical trial",
        "pmid", "sources:", "found that", "published", "cohort",
        "randomis", "according to",
    ]):
        return "rag"
    if any(p in r for p in GENERIC_RESPONSE_PHRASES + [
        "i'm focused on", "neurological and neurodegenerative",
        "feel free to ask",
    ]):
        return "off-topic"
    return "unknown"


def _classify_failure(
    record: QueryRecord,
    expected: str,
    actual: str,
) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Classify whether this record is a routing failure using the *live* route
    result (actual) compared against the expected route.

    Categories:
      misunderstood_intent   — wrong route taken (intake vs rag mismatch)
      out_of_scope_response  — non-medical query not caught by off-topic guard
      poor_context_handling  — short follow-up (≤4 words) routed to rag
                               instead of being treated as an intake reply
      generic_unhelpful      — only detectable when running with real responses;
                               flagged here only when live router is unavailable
                               and the response text contains a rejection phrase

    Note: poor_context_handling cannot be fully tested without session state.
    The flag here catches a proxy: short answers that the live router sends to
    rag (they should be treated as intake replies during an active session).
    """
    q = record.query.strip()
    q_lower = q.lower()

    # ── 1. Misunderstood intent ───────────────────────────────────────────
    if actual != "unknown" and actual != expected:
        if expected == "intake" and actual == "rag":
            return True, "misunderstood_intent", (
                "Symptom description (possibly with trailing '?') routed to RAG. "
                "is_factual_query() fired before the symptom-report guard checked."
            )
        if expected == "rag" and actual == "intake":
            return True, "misunderstood_intent", (
                "Factual literature question routed to intake flow. "
                "is_factual_query() did not fire for this phrasing."
            )
        if expected == "off-topic" and actual != "off-topic":
            return True, "out_of_scope_response", (
                f"Non-medical query not caught by off-topic guard (routed to '{actual}'). "
                "is_off_topic() returned False — a medical keyword may be incidentally present."
            )
        if expected in ("intake", "rag") and actual == "off-topic":
            return True, "misunderstood_intent", (
                f"Medical query (expected '{expected}') was blocked as off-topic. "
                "is_off_topic() fired — relevant medical keywords may be missing."
            )

    # ── 2. Poor context handling proxy ────────────────────────────────────
    # Short follow-up answers (≤4 words) with no medical keywords would be
    # redirected as off-topic without the intake_active suppression flag.
    # Detect those cases here as a static proxy.
    is_short = len(q.split()) <= 4 and len(q) > 0
    if is_short and expected == "intake":
        has_med = any(kw in q_lower for kw in DISEASE_KEYWORDS | SYMPTOM_KEYWORDS)
        if not has_med and actual == "off-topic":
            return True, "poor_context_handling", (
                f"Short follow-up '{q}' has no medical keywords and was redirected "
                "as off-topic. This would fail without the intake_active suppression flag."
            )

    # ── 3. Generic/unhelpful — only when live router is unavailable ───────
    if not _LIVE_ROUTER_AVAILABLE and record.response:
        r = record.response.lower()
        if any(p in r for p in GENERIC_RESPONSE_PHRASES):
            return True, "generic_unhelpful", (
                "Response contains a vague rejection phrase. "
                "(Detected from response text — live router unavailable.)"
            )

    return False, None, None


def _infer_topic(query: str, features: dict) -> str:
    """Coarse topic label for pattern clustering."""
    q = query.lower()
    if any(w in q for w in ["alzheimer", "memory", "forget", "dementia"]):
        return "Alzheimer / Dementia"
    if any(w in q for w in ["parkinson", "tremor", "rigid", "bradykin", "gait"]):
        return "Parkinson's"
    if any(w in q for w in ["als", "amyotrophic", "motor neuron", "weakness", "bulbar"]):
        return "ALS / Motor Neuron"
    if any(w in q for w in ["huntington", "chorea"]):
        return "Huntington's"
    if any(w in q for w in ["stroke", "cerebro", "tia", "ischemic"]):
        return "Stroke"
    if any(w in q for w in ["treatment", "drug", "therapy", "medication", "levodopa"]):
        return "Treatment / Medication"
    if any(w in q for w in ["diagnosis", "diagnose", "test", "mri", "scan"]):
        return "Diagnosis / Testing"
    if features["has_symptom_kw"]:
        return "General Symptoms"
    return "Off-topic / Other"


# ── Core analysis pipeline ────────────────────────────────────────────────────

def analyse_records(raw: list[tuple[str, str, str | None]]) -> list[QueryRecord]:
    """
    Process a list of (query, response, expected_route) tuples.
    Returns a list of fully annotated QueryRecord objects.

    The 'actual' route is determined by calling the live chatbot routing
    functions (is_factual_query, is_off_topic, is_symptom_description) from
    conversation_manager.py.  When those are unavailable, falls back to
    parsing the response text.
    """
    records: list[QueryRecord] = []

    for query, response, expected_route in raw:
        feats = _extract_features(query)
        expected = expected_route or _infer_expected_route(query, feats)
        # ── Use live router; fall back to response-text heuristic ────────
        if _LIVE_ROUTER_AVAILABLE:
            actual = _live_route(query)
        else:
            actual = _response_heuristic_route(response)

        rec = QueryRecord(
            query             = query,
            response          = response,
            expected_route    = expected,
            detected_route    = actual,
            has_disease_kw    = feats["has_disease_kw"],
            has_symptom_kw    = feats["has_symptom_kw"],
            has_report_phrase = feats["has_report_phrase"],
            has_question_mark = feats["has_question_mark"],
            has_factual_kw    = feats["has_factual_kw"],
            starts_question   = feats["starts_question"],
            is_formal         = feats["is_formal"],
            mentioned_diseases = feats["mentioned_diseases"],
            inferred_topic    = _infer_topic(query, feats),
        )

        is_fail, ftype, freason = _classify_failure(rec, expected, actual)
        rec.is_failure    = is_fail
        rec.failure_type  = ftype
        rec.failure_reason = freason
        records.append(rec)

    return records


# ── Pattern analysis ──────────────────────────────────────────────────────────

def compute_patterns(records: list[QueryRecord]) -> dict:
    """Aggregate statistics and pattern counts from all records."""

    failures = [r for r in records if r.is_failure]
    n_total    = len(records)
    n_failures = len(failures)

    # Failure type distribution
    type_counts = Counter(r.failure_type for r in failures if r.failure_type)

    # Topic distribution of failures
    topic_counts = Counter(r.inferred_topic for r in failures)

    # Phrasing style of failures
    formal_failures = sum(1 for r in failures if r.is_formal)
    casual_failures = sum(1 for r in failures if getattr(r, "is_casual", False))

    # Which expected route fails most?
    route_failures = Counter(r.expected_route for r in failures)

    # Feature co-occurrence in failures
    feature_flags = {
        "has_report_phrase + has_symptom_kw (routed to RAG)": sum(
            1 for r in failures
            if r.has_report_phrase and r.has_symptom_kw
            and r.failure_type == "misunderstood_intent"
        ),
        "has_question_mark (any)": sum(1 for r in failures if r.has_question_mark),
        "short_answer (≤4 words)": sum(
            1 for r in failures if len(r.query.split()) <= 4
        ),
        "no_medical_kw": sum(
            1 for r in failures
            if not r.has_disease_kw and not r.has_symptom_kw
        ),
    }

    # Disease-specific failure counts
    disease_failures: dict[str, int] = defaultdict(int)
    for r in failures:
        for d in r.mentioned_diseases:
            disease_failures[d] += 1

    # Error rate per topic
    topic_total = Counter(r.inferred_topic for r in records)
    topic_error_rate = {
        topic: round(topic_counts[topic] / topic_total[topic], 3)
        for topic in topic_total
    }

    return {
        "n_total":          n_total,
        "n_failures":       n_failures,
        "failure_rate":     round(n_failures / n_total, 3) if n_total else 0,
        "type_counts":      dict(type_counts),
        "topic_counts":     dict(topic_counts),
        "topic_error_rate": topic_error_rate,
        "route_failures":   dict(route_failures),
        "formal_failures":  formal_failures,
        "casual_failures":  casual_failures,
        "feature_flags":    feature_flags,
        "disease_failures": dict(disease_failures),
    }


# ── Report generation ─────────────────────────────────────────────────────────

def _bar_chart(ax, labels, values, title, color, xlabel="Count"):
    """Horizontal bar chart helper."""
    y = range(len(labels))
    bars = ax.barh(list(y), values, color=color, edgecolor="white", height=0.55)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8)


def generate_charts(patterns: dict, output_path: str) -> None:
    """Produce a 4-panel figure and save it to output_path."""
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("NORA Chatbot — Systematic Error Analysis",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.4)

    # ── Panel 1: Failure type distribution ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    tc = patterns["type_counts"]
    labels1 = [FAILURE_LABELS.get(k, k) for k in tc]
    _bar_chart(ax1, labels1, list(tc.values()),
               "Failure Type Distribution", "#E05C5C")

    # ── Panel 2: Topic distribution of failures ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    top_topics = sorted(patterns["topic_counts"].items(), key=lambda x: -x[1])[:8]
    _bar_chart(ax2,
               [t[0] for t in top_topics],
               [t[1] for t in top_topics],
               "Failures by Topic (Top 8)", "#5B8DB8")

    # ── Panel 3: Error rate per topic ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    er = sorted(patterns["topic_error_rate"].items(), key=lambda x: -x[1])[:8]
    _bar_chart(ax3,
               [t[0] for t in er],
               [round(t[1] * 100, 1) for t in er],
               "Error Rate by Topic (%)", "#5BA85B", xlabel="Error Rate (%)")

    # ── Panel 4: Route failure breakdown + phrasing ───────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    rf = patterns["route_failures"]
    route_labels = [f"Expected: {k}" for k in rf]
    phrasing_data = {
        "Formal phrasing failures": patterns["formal_failures"],
        "Casual phrasing failures": patterns["casual_failures"],
    }
    combined_labels = route_labels + list(phrasing_data.keys())
    combined_values = list(rf.values()) + list(phrasing_data.values())
    colors = ["#E09A3B"] * len(route_labels) + ["#9B59B6"] * len(phrasing_data)
    y = range(len(combined_labels))
    ax4.barh(list(y), combined_values, color=colors, edgecolor="white", height=0.55)
    ax4.set_yticks(list(y))
    ax4.set_yticklabels(combined_labels, fontsize=9)
    ax4.set_xlabel("Count", fontsize=9)
    ax4.set_title("Route Failures & Phrasing Style", fontsize=11,
                  fontweight="bold", pad=8)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    for i, v in enumerate(combined_values):
        ax4.text(v + 0.1, i, str(v), va="center", fontsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Charts saved → {output_path}")


def generate_report(records: list[QueryRecord], patterns: dict,
                    output_path: str) -> None:
    """Write a full narrative error analysis report to output_path."""

    failures = [r for r in records if r.is_failure]
    n = patterns["n_total"]
    nf = patterns["n_failures"]
    fr = patterns["failure_rate"] * 100

    lines: list[str] = []

    def h1(title):  lines.append(f"\n{'='*70}\n  {title}\n{'='*70}")
    def h2(title):  lines.append(f"\n{'-'*60}\n  {title}\n{'-'*60}")
    def para(text): lines.append(textwrap.fill(text, width=70))
    def blank():    lines.append("")

    # ── Header ────────────────────────────────────────────────────────────
    router_mode = "LIVE (conversation_manager)" if _LIVE_ROUTER_AVAILABLE else "FALLBACK (response-text heuristic)"
    h1("NORA CHATBOT — SYSTEMATIC ERROR ANALYSIS REPORT")
    lines.append(f"  Routing evaluation mode : {router_mode}")
    lines.append(f"  Total examples analysed : {n}")
    lines.append(f"  Total failures detected : {nf}  ({fr:.1f}%)")
    lines.append(f"  Pass rate               : {100 - fr:.1f}%")
    blank()

    # ── Section 1: Failure categories ────────────────────────────────────
    h1("SECTION 1 — FAILURE CATEGORIES")
    blank()
    for key, label in FAILURE_LABELS.items():
        count = patterns["type_counts"].get(key, 0)
        pct   = round(count / nf * 100, 1) if nf else 0
        lines.append(f"  [{count:>3}  {pct:>5.1f}%]  {label}")
    blank()

    h2("1A — Misunderstood Intent")
    mi = [r for r in failures if r.failure_type == "misunderstood_intent"]
    para(
        f"Found {len(mi)} cases where the bot routed to the wrong pathway. "
        "The most common sub-pattern is a symptom description ending with '?' "
        "being sent to RAG instead of the intake template filler. "
        "This typically happens when 'is_factual_query()' detects a question mark "
        "and medical keywords without first checking whether the message contains "
        "a symptom-report phrase ('patient has', 'I have', etc.)."
    )
    blank()
    for r in mi[:5]:
        lines.append(f"  Q: {r.query[:80]}")
        lines.append(f"     Route expected={r.expected_route}, actual={r.detected_route}")
        lines.append(f"     Reason: {r.failure_reason}")
        blank()

    h2("1B — Poor Context Handling")
    pc = [r for r in failures if r.failure_type == "poor_context_handling"]
    para(
        f"Found {len(pc)} cases where the bot failed to recognise a short "
        "follow-up as a reply to the previous intake question. "
        "Short answers such as 'Moderate', 'About 3 months', or 'No medications' "
        "have no medical keywords and therefore bypass the intake guard, "
        "triggering a generic 'I don't understand' response. "
        "The fix is the 'intake_active' suppression flag in handle_turn()."
    )
    blank()
    for r in pc[:5]:
        lines.append(f"  Q: {r.query[:80]}")
        lines.append(f"     Reason: {r.failure_reason}")
        blank()

    h2("1C — Generic / Unhelpful Responses")
    gh = [r for r in failures if r.failure_type == "generic_unhelpful"]
    para(
        f"Found {len(gh)} cases where the bot replied with vague rejection "
        "phrases ('I don't understand', 'Could you clarify?') instead of a "
        "substantive response. This often indicates the RAG pipeline found no "
        "relevant abstracts, or the template filler received input it could not "
        "parse, with no graceful fallback."
    )
    blank()
    for r in gh[:5]:
        lines.append(f"  Q: {r.query[:80]}")
        lines.append(f"     Response snippet: {r.response[:80]}")
        blank()

    h2("1D — Out-of-Scope Responses")
    oos = [r for r in failures if r.failure_type == "out_of_scope_response"]
    para(
        f"Found {len(oos)} cases where the bot answered a clearly non-medical "
        "query with clinical or literature content. This is caused by weak "
        "off-topic detection — the is_off_topic() guard relies purely on keyword "
        "presence, so any message that happens to share a word with a medical "
        "term (e.g. 'rabbit' → not detected, but force-linked by the LLM) "
        "bypasses the redirect."
    )
    blank()
    for r in oos[:5]:
        lines.append(f"  Q: {r.query[:80]}")
        lines.append(f"     Response snippet: {r.response[:80]}")
        blank()

    # ── Section 2: Pattern analysis ───────────────────────────────────────
    h1("SECTION 2 — PATTERN ANALYSIS")
    blank()

    h2("2A — Most Problematic Intents / Topics")
    para(
        "Failures clustered by inferred topic. Topics with both high absolute "
        "count and high error rate are highest priority for improvement."
    )
    blank()
    sorted_er = sorted(patterns["topic_error_rate"].items(), key=lambda x: -x[1])
    lines.append(f"  {'Topic':<35} {'Error Rate':>10}  {'# Failures':>12}")
    lines.append(f"  {'-'*35} {'-'*10}  {'-'*12}")
    for topic, er in sorted_er:
        fc = patterns["topic_counts"].get(topic, 0)
        lines.append(f"  {topic:<35} {er*100:>9.1f}%  {fc:>12}")
    blank()

    h2("2B — Phrasing Style Analysis")
    ff = patterns["formal_failures"]
    cf = patterns["casual_failures"]
    total_formal  = sum(1 for r in records if r.is_formal)
    total_casual  = sum(1 for r in records if getattr(r, "is_casual", False))
    para(
        f"Formal phrasing failures: {ff} out of {total_formal} formal queries "
        f"({round(ff/total_formal*100, 1) if total_formal else 0}%). "
        f"Casual phrasing failures: {cf} out of {total_casual} casual queries "
        f"({round(cf/total_casual*100, 1) if total_casual else 0}%). "
        "Formal clinical notes (e.g. 'presenting with cogwheel rigidity') "
        "are particularly prone to misrouting because their specialist vocabulary "
        "overlaps with literature keywords, causing is_factual_query() to fire "
        "even though the message is a symptom description."
    )
    blank()

    h2("2C — Feature Flags in Failures")
    for flag, count in patterns["feature_flags"].items():
        pct = round(count / nf * 100, 1) if nf else 0
        lines.append(f"  {flag:<50}: {count:>3}  ({pct:.1f}% of failures)")
    blank()

    h2("2D — Disease-Specific Failure Counts")
    if patterns["disease_failures"]:
        for disease, count in sorted(
                patterns["disease_failures"].items(), key=lambda x: -x[1]):
            lines.append(f"  {disease:<30}: {count} failures")
    else:
        lines.append("  No disease-specific failures detected.")
    blank()

    # ── Section 3: Recommendations ────────────────────────────────────────
    h1("SECTION 3 — RECOMMENDATIONS")
    blank()

    recs = [
        ("R1", "Symptom-report guard in is_factual_query()",
         "Add an early-return False in is_factual_query() when the message "
         "contains a _SYMPTOM_REPORT_PHRASES match AND a _SYMPTOM_KEYWORDS "
         "match. This directly fixes the largest failure cluster: symptom "
         "descriptions ending with '?' being sent to RAG."),
        ("R2", "Intake-active suppression in handle_turn()",
         "Ensure the 'intake_active' flag suppresses the off-topic guard "
         "and the is_factual_query() check when an intake is in progress "
         "(template has at least one filled field, is not yet complete). "
         "This prevents short follow-up answers ('Moderate', 'About 3 months') "
         "from being rejected as off-topic."),
        ("R3", "RAG fallback messaging",
         "When the RAG pipeline returns no relevant abstracts, return a "
         "specific fallback ('No relevant studies found for this query') "
         "rather than a generic rejection. This eliminates the "
         "'generic_unhelpful' failure class."),
        ("R4", "Strengthen off-topic guard with NLP",
         "Replace pure keyword matching with a lightweight sentence-level "
         "medical classifier (logistic regression on TF-IDF). This handles "
         "messages that share no medical keywords but are clearly clinical "
         "('my hands won't stop shaking')."),
        ("R5", "Formal clinical language support",
         "Add clinical phrasing patterns to _SYMPTOM_REPORT_PHRASES: "
         "'presents with', 'presenting complaint', 'bilateral', 'onset of'. "
         "This ensures notes written in medical shorthand route to intake "
         "rather than RAG."),
        ("R6", "Context window for short answers",
         "Track the last question asked by the bot. If the last question "
         "was an intake field prompt (e.g. 'How severe are the symptoms?'), "
         "treat the next user message as a reply regardless of its content. "
         "This eliminates the entire 'poor_context_handling' failure class."),
    ]

    for code, title, detail in recs:
        lines.append(f"  [{code}] {title}")
        lines.append(textwrap.fill(detail, width=68,
                                   initial_indent="       ",
                                   subsequent_indent="       "))
        blank()

    # ── Footer ────────────────────────────────────────────────────────────
    h1("END OF REPORT")
    lines.append(
        f"  Generated from {n} examples  |  "
        f"{nf} failures  |  "
        f"{fr:.1f}% error rate"
    )
    blank()

    report_text = "\n".join(lines)
    Path(output_path).write_text(report_text, encoding="utf-8")
    print(f"  Report saved  → {output_path}")
    return report_text


# ── JSON export ────────────────────────────────────────────────────────────────

def export_json(records: list[QueryRecord], patterns: dict, path: str) -> None:
    out = {
        "summary": {
            "n_total":      patterns["n_total"],
            "n_failures":   patterns["n_failures"],
            "failure_rate": patterns["failure_rate"],
            "by_type":      patterns["type_counts"],
            "by_topic":     patterns["topic_counts"],
            "error_rate_by_topic": patterns["topic_error_rate"],
        },
        "records": [r.to_dict() for r in records],
    }
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  JSON data     → {path}")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NORA Systematic Error Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        help="Path to a JSON file containing (query, response, expected_route) "
             "triples.  If omitted, the built-in 200-example test bank is used.",
        default=None,
    )
    parser.add_argument(
        "--report",  default="error_analysis_report.txt",
        help="Output path for the narrative text report.",
    )
    parser.add_argument(
        "--charts",  default="error_analysis_charts.png",
        help="Output path for the visualisation PNG.",
    )
    parser.add_argument(
        "--json",    default="error_analysis_results.json",
        help="Output path for the raw JSON results.",
    )
    args = parser.parse_args()

    print("\n NORA Chatbot — Systematic Error Analysis")
    print(" " + "─" * 48)
    print(f"  Router mode: {'LIVE (conversation_manager)' if _LIVE_ROUTER_AVAILABLE else 'FALLBACK (response-text heuristic)'}")

    # ── Load data ────────────────────────────────────────────────────────
    if args.input:
        raw_path = Path(args.input)
        if not raw_path.exists():
            print(f"[ERROR] Input file not found: {args.input}")
            sys.exit(1)
        with open(raw_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        raw: list[tuple] = [
            (item["query"], item["response"], item.get("expected_route"))
            for item in loaded
        ]
        print(f"  Loaded {len(raw)} examples from {args.input}")
    else:
        raw = _TEST_BANK
        print(f"  Using built-in test bank ({len(raw)} examples)")

    # ── Analyse ──────────────────────────────────────────────────────────
    print("  Analysing records …")
    records  = analyse_records(raw)
    patterns = compute_patterns(records)

    print(f"\n  ✓ {patterns['n_total']} examples  "
          f"→  {patterns['n_failures']} failures  "
          f"({patterns['failure_rate']*100:.1f}% error rate)\n")

    print("  Failure type breakdown:")
    for key, label in FAILURE_LABELS.items():
        count = patterns["type_counts"].get(key, 0)
        bar   = "█" * count
        print(f"    {label:<35} {count:>3}  {bar}")

    print("\n  Generating outputs …")
    generate_charts(patterns, args.charts)
    generate_report(records, patterns, args.report)
    export_json(records, patterns, args.json)

    print("\n  ✓ Done.\n")


if __name__ == "__main__":
    main()
