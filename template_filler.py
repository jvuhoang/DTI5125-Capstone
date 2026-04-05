"""
template_filler.py — Clinical Intake Template + Slot Filler
============================================================
Defines the ClinicalTemplate dataclass (6 fields) and the extraction
logic that populates it from free-text user messages using:
  - symptom_synonyms  → colloquial / paraphrase lookup (FIRST pass)
  - scispaCy NER      → formal biomedical entity recognition (SECOND pass)
  - regex             → duration, age/gender, severity
  - keyword rules     → family history

Fields and priority order (symptoms asked first):
  1. primary_symptoms    — what the patient is experiencing
  2. duration            — how long symptoms have been present
  3. severity            — mild / moderate / severe
  4. age_gender          — patient demographics
  5. family_history      — neurological family history
  6. current_medications — drugs currently being taken

The template is considered "scoreable" once symptoms + duration + severity
are all filled — at that point the symptom scorer is triggered.
"""

from dataclasses import dataclass
from typing import Optional
import re
from difflib import get_close_matches
from symptom_synonyms import lookup_symptoms

# Load scispaCy model once at module import — fully optional.
# On Streamlit Cloud spacy is not installed (scipy conflict), so we wrap
# the import itself in try/except, not just the model load.
try:
    import spacy
    _nlp = spacy.load("en_ner_bc5cdr_md")
except (ImportError, OSError):
    _nlp = None
    print(
        "[WARN] template_filler: spacy / en_ner_bc5cdr_md not available. "
        "NER-based extraction disabled — falling back to synonym lookup.\n"
        "Install locally with:\n"
        "  pip install scispacy==0.5.4 spacy==3.7.4\n"
        "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
        "releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
    )


# ── Template dataclass ────────────────────────────────────────────────────────

@dataclass
class ClinicalTemplate:
    age_gender:          Optional[str] = None
    primary_symptoms:    Optional[str] = None
    duration:            Optional[str] = None
    severity:            Optional[str] = None   # "mild" | "moderate" | "severe"
    family_history:      Optional[str] = None
    current_medications: Optional[str] = None

    def filled_count(self) -> int:
        """Number of fields that have been filled (excludes pending placeholders)."""
        return sum(1 for v in vars(self).values()
                   if v is not None and v != "__YES__")

    def is_scoreable(self) -> bool:
        """True when the minimum required fields for disease scoring are present."""
        return (
            self.primary_symptoms is not None
            and self.duration is not None
            and self.severity is not None
        )

    def is_complete(self) -> bool:
        """True when all 6 fields are filled (placeholder __YES__ counts as incomplete)."""
        return all(v is not None and v != "__YES__" for v in vars(self).values())

    def to_text(self) -> str:
        """Flatten the template into a single clinical summary string for the scorer."""
        parts = []
        if self.age_gender:          parts.append(f"Patient: {self.age_gender}")
        if self.primary_symptoms:    parts.append(f"Symptoms: {self.primary_symptoms}")
        if self.duration:            parts.append(f"Duration: {self.duration}")
        if self.severity:            parts.append(f"Severity: {self.severity}")
        if self.family_history:      parts.append(f"Family history: {self.family_history}")
        if self.current_medications: parts.append(f"Medications: {self.current_medications}")
        return ". ".join(parts)

    def to_dict(self) -> dict:
        return {
            "age_gender":          self.age_gender,
            "primary_symptoms":    self.primary_symptoms,
            "duration":            self.duration,
            "severity":            self.severity,
            "family_history":      self.family_history,
            "current_medications": self.current_medications,
        }

    def __repr__(self) -> str:
        filled = self.filled_count()
        return (
            f"ClinicalTemplate({filled}/6 filled | "
            f"scoreable={self.is_scoreable()} | "
            f"complete={self.is_complete()})"
        )


# ── Priority queue and follow-up questions ────────────────────────────────────

FIELD_PRIORITY = [
    "primary_symptoms",
    "duration",
    "severity",
    "age_gender",
    "family_history",
    "current_medications",
]

FOLLOW_UP_QUESTIONS = {
    "primary_symptoms":
        "What symptoms are you or the patient experiencing?",
    "duration":
        "How long have these symptoms been present?",
    "severity":
        "How would you describe the severity — mild, moderate, or severe?",
    "age_gender":
        "What is the patient's age and gender?",
    "family_history":
        "Is there any family history of neurological or neurodegenerative conditions?",
    "current_medications":
        "What medications is the patient currently taking, if any?",
}


def next_question(template: ClinicalTemplate) -> Optional[str]:
    """Return the next follow-up question based on unfilled fields, or None if complete."""
    # Special case: user confirmed family history exists but gave no detail yet
    if template.family_history == "__YES__":
        return (
            "Which neurological condition was it, and which family member was affected? "
            "(e.g. 'my mother had Parkinson's' or 'father had Alzheimer's')"
        )
    for field_name in FIELD_PRIORITY:
        if getattr(template, field_name) is None:
            return FOLLOW_UP_QUESTIONS[field_name]
    return None


# ── Regex patterns ────────────────────────────────────────────────────────────

_DURATION_PATTERNS = [
    # Numeric + unit (most specific first): "3 years", "two weeks"
    r"\b(?:for\s+|about\s+|around\s+|over\s+|nearly\s+|almost\s+|roughly\s+)?\d+\s*(?:and\s+a\s+half\s+)?(?:year|month|week|day)s?\b",
    # Written-out numbers: "two years", "three months"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:year|month|week|day)s?\b",
    # "a week", "a month", "a year"
    r"\ba\s+(?:year|month|week|day)s?\b",
    # "a few weeks", "a few months"
    r"\ba\s+few\s+(?:year|month|week|day)s?\b",
    # "several years", "several months"
    r"\bseveral\s+(?:year|month|week|day)s?\b",
    # "a couple of weeks", "a couple months"
    r"\ba?\s*couple\s+(?:of\s+)?(?:year|month|week|day)s?\b",
    # "many years", "some months"
    r"\b(?:many|some|multiple)\s+(?:year|month|week|day)s?\b",
    # "for the past few months", "for the past year"
    r"\bfor\s+the\s+past\s+(?:\w+\s+){0,2}(?:year|month|week|day)s?\b",
    # "since last year/month/week", "since childhood", "since birth"
    r"\bsince\s+(?:last\s+)?(?:year|month|week|childhood|birth|early\s+\w+)\b",
    # "a long time", "a long while", "long time"
    r"\b(?:a\s+)?(?:very\s+)?long\s+(?:time|while)\b",
    # "a while", "quite a while"
    r"\b(?:quite\s+)?a\s+while\b",
    # "years and years", "months now"
    r"\b(?:year|month|week|day)s\s+(?:and\s+(?:year|month|week|day)s\s+)?(?:now|already)?\b",
    # Vague time: "recently", "just started", "new symptom"
    r"\b(?:recently|lately|just\s+(?:started|noticed|begun|developed)|new\s+symptom|not\s+long)\b",
    # "getting worse over time", "ongoing for"
    r"\b(?:ongoing|progressing|worsening|getting\s+worse)\b",
]

# Matches: "65 year old", "65yo", "65 y.o.", "65 male", "65 M", "65M",
#          "male 65", "M 65", "45 F", "45F", "34 female", "F 34"
_AGE_PATTERN = re.compile(
    r"\b(\d{2,3})[- ]*(?:year[s]?[- ]*old|yo|y\.o\.?|male|female|m|f)\b"
    r"|\b(?:male|female|m|f)[- ]*(\d{2,3})\b",
    re.I
)
_MALE_WORDS   = {"male", "man", "gentleman", "he", "him", "his", "boy"}
_FEMALE_WORDS = {"female", "woman", "lady", "she", "her", "hers", "girl"}
_SEVERITY_WORDS = {"mild", "moderate", "severe", "light", "serious", "intense", "bad"}
_FAMILY_WORDS   = {
    "family", "mother", "father", "parent", "sibling", "brother", "sister",
    "grandmother", "grandfather", "grandparent", "aunt", "uncle", "cousin",
    "hereditary", "genetic", "inherited", "runs in the family", "No", "no", "Not", "no"
}


# ── Main extraction function ──────────────────────────────────────────────────

_FUZZY_VOCAB = [
    # Time units — corrects "yera", "yaer", "monht", "weeak", etc.
    "year", "years", "month", "months", "week", "weeks", "day", "days",
    # Severity — corrects "seveer", "miled", "moderrate", etc.
    "mild", "moderate", "severe",
    # Common yes/no variants
    "yes", "no", "none",
]


def _fuzzy_correct(text: str) -> str:
    """
    Replace unrecognised short tokens with their closest known word.
    Only corrects tokens that are 3+ characters and have exactly one
    close match (cutoff 0.75) to avoid false substitutions.
    """
    tokens = text.split()
    corrected = []
    for token in tokens:
        # Strip trailing punctuation for matching, reattach afterward
        stripped = token.rstrip(".,!?;:")
        suffix   = token[len(stripped):]
        if len(stripped) >= 3 and stripped not in _FUZZY_VOCAB:
            matches = get_close_matches(stripped, _FUZZY_VOCAB,
                                        n=1, cutoff=0.75)
            if matches:
                corrected.append(matches[0] + suffix)
                continue
        corrected.append(token)
    return " ".join(corrected)


def extract_from_text(user_text: str, template: ClinicalTemplate) -> ClinicalTemplate:
    """
    Parse a user message and update any unfilled template fields.
    Uses scispaCy NER for medical entities and regex for structured fields.
    Modifies the template in place and returns it.
    """
    # Apply fuzzy spelling correction before any field extraction
    user_text  = _fuzzy_correct(user_text)
    text_lower = user_text.lower()
    words      = set(text_lower.split())

    # ── Severity ──────────────────────────────────────────────────────────────
    if template.severity is None:
        for sev in ["severe", "moderate", "mild"]:   # check severe first
            if sev in text_lower:
                template.severity = sev
                break
        if template.severity is None:
            severe_phrases = {
                "bad", "serious", "intense", "extreme", "very bad", "really bad",
                "unbearable", "debilitating", "crippling", "can't function",
                "can't walk", "can't talk", "getting worse", "worsening",
                "significantly", "drastically", "a lot", "quite bad", "pretty bad",
                "affecting daily", "interfering with", "constant", "all the time",
            }
            mild_phrases = {
                "light", "slight", "little", "minor", "manageable", "not too bad",
                "barely", "occasionally", "sometimes", "comes and goes", "on and off",
                "not that bad", "liveable", "livable", "tolerable", "okay",
            }
            moderate_phrases = {
                "moderate", "medium", "noticeable", "bothersome", "concerning",
                "bit worse", "somewhat", "fairly", "rather", "getting there",
            }
            if any(p in text_lower for p in severe_phrases):
                template.severity = "severe"
            elif any(p in text_lower for p in moderate_phrases):
                template.severity = "moderate"
            elif any(p in text_lower for p in mild_phrases):
                template.severity = "mild"

    # ── Duration ─────────────────────────────────────────────────────────────
    if template.duration is None:
        for pat in _DURATION_PATTERNS:
            m = re.search(pat, text_lower, re.IGNORECASE)
            if m:
                template.duration = m.group().strip()
                break
        # Last-resort: if they wrote anything that sounds like time passing
        if template.duration is None:
            vague = re.search(
                r"\b(ages|forever|years?|months?|weeks?|days?)\b", text_lower
            )
            if vague:
                template.duration = vague.group().strip()

    # ── Age / gender ─────────────────────────────────────────────────────────
    if template.age_gender is None:
        age_m = _AGE_PATTERN.search(text_lower)
        if age_m:
            # Group 1 = age-first ("65 male"), group 2 = gender-first ("M 65")
            age = age_m.group(1) or age_m.group(2)
            matched = age_m.group(0).lower()
            msg_words = set(text_lower.split())
            # Check the matched token first (catches "65M", "65 F", "M 65", etc.)
            # then fall back to scanning the full message for gender words
            if any(g in matched for g in ("male", " m", "m ")):
                gender = "male"
            elif any(g in matched for g in ("female", " f", "f ")):
                gender = "female"
            elif msg_words & _MALE_WORDS:
                gender = "male"
            elif msg_words & _FEMALE_WORDS:
                gender = "female"
            else:
                gender = "unknown gender"
            template.age_gender = f"{age} years old, {gender}"

    # ── Family history ────────────────────────────────────────────────────────
    _AFFIRM = {"yes", "yeah", "yep", "yup", "sure", "correct", "there is",
               "there are", "there was", "we do", "i do"}
    _DENY   = {"no", "none", "nope", "nah", "never", "not", "n/a", "unknown"}

    if template.family_history == "__YES__":
        # User previously said "yes" — now capture the actual detail
        disease_keywords = [
            "alzheimer", "parkinson", "huntington", "als", "dementia",
            "stroke", "neurological", "memory", "tremor",
        ]
        relative_keywords = [
            "mother", "father", "parent", "sibling", "brother", "sister",
            "grandmother", "grandfather", "grandparent", "aunt", "uncle",
            "cousin", "relative", "family member",
        ]
        has_disease  = any(kw in text_lower for kw in disease_keywords)
        has_relative = any(kw in text_lower for kw in relative_keywords)
        if has_disease or has_relative:
            template.family_history = user_text.strip()[:150]
        elif text_lower.strip() in _DENY or any(
            p in text_lower for p in ["no condition", "not sure", "don't know",
                                      "unknown", "unsure"]
        ):
            template.family_history = "yes — condition not specified"

    elif template.family_history is None:
        # Bare affirmative → set placeholder so we can ask for detail next turn
        if text_lower.strip() in _AFFIRM or any(
            p in text_lower for p in ["yes there", "yes, there", "yes my",
                                      "yes his", "yes her", "yes their"]
        ):
            template.family_history = "__YES__"
        # Bare negative → mark as none known
        elif text_lower.strip() in _DENY or any(
            p in text_lower for p in ["no family", "no history", "none known",
                                      "not that i know", "no neurological"]
        ):
            template.family_history = "none known"
        # Substantive answer with family/disease keywords → store directly
        elif words & _FAMILY_WORDS or any(p in text_lower for p in [
            "runs in the family", "family history", "genetic condition"
        ]):
            template.family_history = user_text.strip()[:150]

    # ── Disease-name blocklist — never add a diagnosis name as a symptom ─────────
    # scispaCy NER correctly tags disease names like "Parkinson's Disease" as
    # DISEASE entities, but they describe a diagnosis, not a symptom.  We filter
    # them out so they don't pollute primary_symptoms.
    _DISEASE_BLOCKLIST = {
        "alzheimer's disease", "alzheimer's", "alzheimer disease", "alzheimer",
        "parkinson's disease", "parkinson's", "parkinson disease", "parkinson",
        "parkinsonism", "als", "amyotrophic lateral sclerosis",
        "motor neuron disease", "lou gehrig's disease", "mnd",
        "huntington's disease", "huntington's", "huntington disease", "huntington",
        "dementia", "mild cognitive impairment", "vascular dementia",
        "lewy body dementia", "frontotemporal dementia",
        "stroke", "cerebrovascular accident", "tia",
        "multiple sclerosis", "ms", "neurodegenerative disease",
    }

    # ── Symptom extraction — three-pass pipeline (always runs, accumulates) ──────
    # Run on every turn so symptoms mentioned in later messages are also captured.

    new_symptoms = []

    # Pass 1: synonym / paraphrase lookup (catches colloquial speech)
    # e.g. "I lose memory", "my hands shake", "I forget things"
    synonym_hits = lookup_symptoms(user_text)
    new_symptoms.extend(synonym_hits)

    # Pass 2: scispaCy NER (catches formal biomedical terms in the text)
    # Filter out bare disease/condition names — we want symptoms, not diagnoses.
    if _nlp is not None:
        doc          = _nlp(user_text)
        disease_ents = [
            ent.text for ent in doc.ents
            if ent.label_ == "DISEASE"
            and ent.text.lower().strip() not in _DISEASE_BLOCKLIST
        ]
        new_symptoms.extend(disease_ents)

    # Pass 3: keyword fallback (only when passes 1+2 found nothing this turn)
    if not new_symptoms:
        keyword_map = {
            "tremor":           "tremor",
            "memory":           "memory loss",
            "forget":           "memory loss",
            "confused":         "confusion",
            "confusion":        "confusion",
            "weak":             "muscle weakness",
            "weakness":         "muscle weakness",
            "stiff":            "rigidity",
            "stiffness":        "rigidity",
            "seizure":          "seizure",
            "headache":         "headache",
            "dizzy":            "dizziness",
            "dizziness":        "dizziness",
            "fatigue":          "fatigue",
            "tired":            "fatigue",
            "hallucin":         "hallucinations",
            "speech":           "speech difficulty",
            "swallow":          "swallowing difficulty",
            "balance":          "balance problems",
            "coordination":     "coordination problems",
            "rigidity":         "rigidity",
            "bradykinesia":     "bradykinesia",
            "chorea":           "chorea",
            "dementia":         "dementia",
            "shake":            "tremor",
            "shaking":          "tremor",
            "numb":             "numbness",
            "tingle":           "tingling",
            "paralys":          "paralysis",
            "depress":          "depression",
            "anxiety":          "anxiety",
        }
        for kw, canonical in keyword_map.items():
            if kw in text_lower and canonical not in new_symptoms:
                new_symptoms.append(canonical)

    # Merge with existing symptoms — deduplicate, preserve order
    if new_symptoms:
        existing_lower = set()
        if template.primary_symptoms:
            existing_lower = {s.strip().lower() for s in template.primary_symptoms.split(",")}

        truly_new = []
        seen = set()
        for s in new_symptoms:
            key = s.strip().lower()
            if key not in existing_lower and key not in seen:
                truly_new.append(s.strip())
                seen.add(key)

        if truly_new:
            if template.primary_symptoms:
                template.primary_symptoms += ", " + ", ".join(truly_new)
            else:
                template.primary_symptoms = ", ".join(truly_new)

    # ── Medications — scispaCy CHEMICAL entities ──────────────────────────────
    # Only process this field when family_history is already filled, i.e. the
    # chatbot has already asked the medications question.  Without this guard,
    # a bare "no" answering the family-history question would simultaneously
    # set current_medications = "None" and skip the medications question entirely.
    if template.current_medications is None and template.family_history is not None:
        text_clean = user_text.lower().strip().strip('.,!?')

        negation_keywords = ["no", "none", "nothing", "n/a", "no meds", "no medication",
                             "not taking", "not on any"]
        negation_phrases  = [
            "not using", "not taking", "don't take", "do not take",
            "don't use", "do not use", "not on any", "not that",
            "no medication", "no meds", "none currently",
        ]

        is_negative = (
            text_clean in negation_keywords or
            any(phrase in text_clean for phrase in negation_phrases) or
            # Catch replies that START with "none", "no", or "nothing" even when
            # extra context follows (e.g. "none and my sister has dementia").
            # Using a word-boundary regex so "someone" doesn't false-match.
            bool(re.search(r'^(?:none|no|nothing)\b', text_clean))
        )

        if is_negative:
            template.current_medications = "None"
        elif _nlp is not None:
            doc_med   = _nlp(user_text)
            chem_ents = [ent.text for ent in doc_med.ents if ent.label_ == "CHEMICAL"]
            if chem_ents:
                template.current_medications = ", ".join(dict.fromkeys(chem_ents))

    return template


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    template = ClinicalTemplate()
    print(template)

    test_messages = [
        "My 68-year-old father has been experiencing tremors and rigidity for about 2 years.",
        "The severity is moderate. He also has memory problems.",
        "His mother had Parkinson's disease.",
        "He is currently taking levodopa and ropinirole.",
    ]

    for msg in test_messages:
        print(f"\nInput: '{msg}'")
        template = extract_from_text(msg, template)
        print(template)
        nq = next_question(template)
        if nq:
            print(f"Next question: {nq}")
        else:
            print("All fields filled!")

    print(f"\nTemplate text: {template.to_text()}")
    print(f"Scoreable: {template.is_scoreable()}")
