"""
symptom_synonyms.py — Synonym & Paraphrase Database for NORA
=============================================================
Maps everyday language, colloquialisms, paraphrases, and clinical terms
onto canonical medical symptom names used by the classifier and RAG system.

Sources merged into this file:
  - Original hand-crafted phrases covering colloquial speech
  - Symptom_entries_en.json (clinical synonym list uploaded by project team)

HOW TO ADD MORE SYNONYMS
-------------------------
Find the right category and add a line:
    "what the user might say": "canonical medical term",

The key must be lowercase. The value should match clinical terminology.

HOW IT IS USED
--------------
template_filler.extract_from_text() calls lookup_symptoms() on EVERY turn,
then accumulates found symptoms. This means multi-symptom messages like
"I have tremors, rigidity, and memory problems" all get captured at once,
and symptoms mentioned across multiple turns are all kept.
"""

import re

# =============================================================================
# PART 1 — EXACT PHRASE LOOKUP
# Keys: lowercase user phrases. Values: canonical medical symptom names.
# Longer / more specific phrases are matched first (sorted by length desc).
# =============================================================================

SYMPTOM_SYNONYMS: dict[str, str] = {

    # ── Resting Tremor (specific subtype) ─────────────────────────────────────
    "resting tremor":               "resting tremor",
    "tremor at rest":               "resting tremor",
    "pill-rolling tremor":          "resting tremor",
    "pill rolling tremor":          "resting tremor",
    "shaking at rest":              "resting tremor",

    # ── Tremor (general) ──────────────────────────────────────────────────────
    "tremor":                       "tremor",
    "tremors":                      "tremor",
    "shaking":                      "tremor",
    "shaky hands":                  "tremor",
    "shaky":                        "tremor",
    "shakes":                       "tremor",
    "hand tremor":                  "tremor",
    "hands shake":                  "tremor",
    "my hands shake":               "tremor",
    "hand shaking":                 "tremor",
    "trembling":                    "tremor",
    "trembling hands":              "tremor",
    "uncontrollable shaking":       "tremor",
    "involuntary shaking":          "tremor",
    "body shakes":                  "tremor",
    "legs shaking":                 "tremor",

    # ── Bradykinesia ──────────────────────────────────────────────────────────
    "bradykinesia":                 "bradykinesia",
    "slowness of movement":         "bradykinesia",
    "slow movements":               "bradykinesia",
    "movement slowness":            "bradykinesia",
    "slow movement":                "bradykinesia",
    "moving slowly":                "bradykinesia",
    "slowness":                     "bradykinesia",
    "slowed down":                  "bradykinesia",
    "movements are slow":           "bradykinesia",
    "taking longer to do things":   "bradykinesia",

    # ── Akinesia ──────────────────────────────────────────────────────────────
    "akinesia":                     "akinesia",
    "absence of movement":          "akinesia",
    "frozen movement":              "akinesia",
    "inability to move":            "akinesia",
    "can't move at all":            "akinesia",
    "freezing":                     "akinesia",
    "freezing episodes":            "akinesia",

    # ── Hypokinesia ───────────────────────────────────────────────────────────
    "hypokinesia":                  "hypokinesia",
    "reduced movement":             "hypokinesia",
    "small movements":              "hypokinesia",
    "diminished movement":          "hypokinesia",

    # ── Rigidity ──────────────────────────────────────────────────────────────
    "rigidity":                     "rigidity",
    "stiffness":                    "rigidity",
    "stiff":                        "rigidity",
    "muscle stiffness":             "rigidity",
    "muscle rigidity":              "rigidity",
    "cogwheel rigidity":            "rigidity",
    "stiff muscles":                "rigidity",
    "rigid muscles":                "rigidity",
    "body feels stiff":             "rigidity",
    "joints stiff":                 "rigidity",
    "tight muscles":                "rigidity",
    "lead pipe rigidity":           "rigidity",

    # ── Postural Instability / Balance ────────────────────────────────────────
    "postural instability":         "balance problems",
    "balance problems":             "balance problems",
    "imbalance":                    "balance problems",
    "poor balance":                 "balance problems",
    "losing balance":               "balance problems",
    "loss of balance":              "balance problems",
    "unsteady on feet":             "balance problems",
    "unsteady":                     "balance problems",

    # ── Gait & Walking ────────────────────────────────────────────────────────
    "gait disturbance":             "gait disturbance",
    "walking problems":             "gait disturbance",
    "shuffling gait":               "gait disturbance",
    "difficulty walking":           "gait disturbance",
    "abnormal gait":                "gait disturbance",
    "trouble walking":              "gait disturbance",
    "shuffling walk":               "gait disturbance",
    "shuffle when walking":         "gait disturbance",
    "can't walk properly":          "gait disturbance",

    # ── Falls ─────────────────────────────────────────────────────────────────
    "falls":                        "falls",
    "frequent falls":               "falls",
    "falling":                      "falls",
    "repeated falls":               "falls",
    "falling over":                 "falls",
    "keep falling":                 "falls",
    "stumbling":                    "falls",

    # ── Limb Weakness ─────────────────────────────────────────────────────────
    "limb weakness":                "limb weakness",
    "arm weakness":                 "limb weakness",
    "leg weakness":                 "limb weakness",
    "weak limbs":                   "limb weakness",
    "weakness in arms or legs":     "limb weakness",
    "arms are weak":                "limb weakness",
    "legs are weak":                "limb weakness",

    # ── Axial Weakness ────────────────────────────────────────────────────────
    "axial weakness":               "axial weakness",
    "trunk weakness":               "axial weakness",
    "neck weakness":                "axial weakness",
    "core weakness":                "axial weakness",

    # ── Muscle Weakness (general) ─────────────────────────────────────────────
    "muscle weakness":              "muscle weakness",
    "muscular weakness":            "muscle weakness",
    "weak muscles":                 "muscle weakness",
    "muscle wasting":               "muscle weakness",
    "weakness":                     "muscle weakness",
    "feeling weak":                 "muscle weakness",
    "can't lift":                   "muscle weakness",
    "difficulty lifting":           "muscle weakness",

    # ── Hypomimia (masked face) ───────────────────────────────────────────────
    "hypomimia":                    "hypomimia",
    "masked face":                  "hypomimia",
    "reduced facial expression":    "hypomimia",
    "facial masking":               "hypomimia",
    "expressionless face":          "hypomimia",
    "face looks blank":             "hypomimia",
    "no facial expression":         "hypomimia",

    # ── Dystonia ──────────────────────────────────────────────────────────────
    "dystonia":                     "dystonia",
    "muscle spasms":                "dystonia",
    "muscle contractions":          "dystonia",
    "abnormal posture":             "dystonia",
    "spasms":                       "dystonia",
    "muscle cramps":                "dystonia",

    # ── Kinesia Paradoxica ────────────────────────────────────────────────────
    "kinesia paradoxica":           "kinesia paradoxica",
    "paradoxical kinesia":          "kinesia paradoxica",

    # ── Dysarthria / Speech ───────────────────────────────────────────────────
    "dysarthria":                   "dysarthria",
    "slurred speech":               "dysarthria",
    "unclear speech":               "dysarthria",
    "speech difficulty":            "dysarthria",
    "difficulty speaking":          "dysarthria",
    "difficulties speaking":        "dysarthria",
    "speech become nasal":          "dysarthria",
    "nasal speech":                 "dysarthria",
    "speech has become nasal":      "dysarthria",
    "trouble speaking":             "dysarthria",
    "hard to talk":                 "dysarthria",
    "slurring":                     "dysarthria",
    "speech problems":              "dysarthria",

    # ── Dysphagia / Swallowing ────────────────────────────────────────────────
    "dysphagia":                    "dysphagia",
    "swallowing difficulty":        "dysphagia",
    "trouble swallowing":           "dysphagia",
    "difficulty swallowing":        "dysphagia",
    "choking on food":              "dysphagia",
    "choke when swallowing":        "dysphagia",
    "can't swallow":                "dysphagia",
    "food going down wrong":        "dysphagia",
    "choking":                      "dysphagia",
    "difficulties swallowing":      "dysphagia",

    # ── Hypophonia (soft voice) ───────────────────────────────────────────────
    "hypophonia":                   "hypophonia",
    "soft voice":                   "hypophonia",
    "quiet voice":                  "hypophonia",
    "low voice volume":             "hypophonia",
    "weak voice":                   "hypophonia",
    "voice getting quieter":        "hypophonia",
    "speaking softly":              "hypophonia",

    # ── Bulbar Dysfunction ────────────────────────────────────────────────────
    "bulbar dysfunction":           "bulbar dysfunction",
    "bulbar symptoms":              "bulbar dysfunction",
    "bulbar palsy":                 "bulbar dysfunction",
    "throat weakness":              "bulbar dysfunction",

    # ── Drooling / Sialorrhoea ────────────────────────────────────────────────
    "sialorrhoea":                  "drooling",
    "drooling":                     "drooling",
    "excessive saliva":             "drooling",
    "drool":                        "drooling",
    "can't control drooling":       "drooling",

    # ── Memory Impairment ─────────────────────────────────────────────────────
    "memory impairment":            "memory loss",
    "episodic memory impairment":   "memory loss",
    "episodic memory loss":         "memory loss",
    "short term memory loss":       "memory loss",
    "forgetting recent events":     "memory loss",
    "memory loss":                  "memory loss",
    "forgetfulness":                "memory loss",
    "poor memory":                  "memory loss",
    "memory problems":              "memory loss",
    "memory problem":               "memory loss",
    "memory issues":                "memory loss",
    "lose memory":                  "memory loss",
    "losing memory":                "memory loss",
    "lost my memory":               "memory loss",
    "i forget things":              "memory loss",
    "i forget stuff":               "memory loss",
    "i keep forgetting":            "memory loss",
    "keep forgetting":              "memory loss",
    "forget things":                "memory loss",
    "forget stuff":                 "memory loss",
    "forgetting things":            "memory loss",
    "forgetting a lot":             "memory loss",
    "forgetful":                    "memory loss",
    "i forget":                     "memory loss",
    "i'm forgetting":               "memory loss",
    "trouble remembering":          "memory loss",
    "hard to remember":             "memory loss",
    "difficulty remembering":       "memory loss",
    "can't remember":               "memory loss",
    "cannot remember":              "memory loss",
    "bad memory":                   "memory loss",
    "short-term memory":            "memory loss",
    "mind going blank":             "memory loss",
    "blanking out":                 "memory loss",
    "can't recall":                 "memory loss",
    "names slip my mind":           "memory loss",
    "losing track":                 "memory loss",
    "don't remember":               "memory loss",
    "not remembering":              "memory loss",

    # ── Cognitive Decline / Reasoning ─────────────────────────────────────────
    "cognitive decline":            "cognitive decline",
    "impaired reasoning":           "cognitive decline",
    "poor judgment":                "cognitive decline",
    "reasoning problems":           "cognitive decline",
    "executive dysfunction":        "cognitive decline",
    "decision making problems":     "cognitive decline",
    "thinking problems":            "cognitive decline",
    "trouble thinking":             "cognitive decline",
    "brain fog":                    "cognitive decline",
    "foggy":                        "cognitive decline",
    "cognitive symptoms":           "cognitive decline",
    "can't concentrate":            "cognitive decline",
    "difficulty concentrating":     "cognitive decline",
    "trouble focusing":             "cognitive decline",
    "can't focus":                  "cognitive decline",

    # ── Language / Aphasia ────────────────────────────────────────────────────
    "language impairment":          "aphasia",
    "word finding difficulty":      "aphasia",
    "aphasia":                      "aphasia",
    "language problems":            "aphasia",
    "difficulty finding words":     "aphasia",
    "trouble finding words":        "aphasia",
    "can't find words":             "aphasia",
    "forget words":                 "aphasia",
    "words come out wrong":         "aphasia",

    # ── Confusion / Disorientation ────────────────────────────────────────────
    "confusion":                    "confusion",
    "disorientation":               "confusion",
    "confused state":               "confusion",
    "mental confusion":             "confusion",
    "getting confused":             "confusion",
    "confused":                     "confusion",
    "disoriented":                  "confusion",
    "not knowing where i am":       "confusion",
    "not knowing what day it is":   "confusion",
    "getting lost":                 "confusion",

    # ── Depression ────────────────────────────────────────────────────────────
    "depression":                   "depression",
    "low mood":                     "depression",
    "sadness":                      "depression",
    "depressive symptoms":          "depression",
    "feeling depressed":            "depression",
    "depressed":                    "depression",
    "feeling down":                 "depression",
    "sad all the time":             "depression",
    "no motivation":                "depression",
    "don't want to do anything":    "depression",

    # ── Anxiety ───────────────────────────────────────────────────────────────
    "anxiety":                      "anxiety",
    "worry":                        "anxiety",
    "anxious":                      "anxiety",
    "nervousness":                  "anxiety",
    "anxiety disorder":             "anxiety",
    "nervous":                      "anxiety",
    "worried all the time":         "anxiety",
    "panic attacks":                "anxiety",

    # ── Hallucinations ────────────────────────────────────────────────────────
    "hallucination":                "hallucinations",
    "hallucinations":               "hallucinations",
    "seeing things":                "hallucinations",
    "visual hallucinations":        "hallucinations",
    "hearing things":               "hallucinations",
    "seeing people that aren't there": "hallucinations",
    "visions":                      "hallucinations",

    # ── Neuropsychiatric / Personality ────────────────────────────────────────
    "neuropsychiatric dysfunction": "personality change",
    "psychiatric disturbance":      "personality change",
    "behavioural changes":          "personality change",
    "personality changes":          "personality change",
    "behavioural symptoms":         "personality change",
    "behaviour changes":            "personality change",
    "personality change":           "personality change",
    "acting differently":           "personality change",
    "not themselves":               "personality change",
    "behavior change":              "personality change",
    "mood swings":                  "personality change",
    "mood changes":                 "personality change",
    "irritable":                    "personality change",

    # ── Pseudobulbar Affect ───────────────────────────────────────────────────
    "pseudobulbar affect":          "pseudobulbar affect",
    "involuntary crying":           "pseudobulbar affect",
    "involuntary laughing":         "pseudobulbar affect",
    "emotional lability":           "pseudobulbar affect",
    "uncontrolled crying":          "pseudobulbar affect",
    "crying for no reason":         "pseudobulbar affect",
    "laughing uncontrollably":      "pseudobulbar affect",

    # ── Akathisia / Restlessness ──────────────────────────────────────────────
    "akathisia":                    "akathisia",
    "restlessness":                 "akathisia",
    "inability to sit still":       "akathisia",
    "inner restlessness":           "akathisia",
    "can't sit still":              "akathisia",
    "restless":                     "akathisia",

    # ── Apathy ────────────────────────────────────────────────────────────────
    "apathy":                       "apathy",
    "no interest":                  "apathy",
    "doesn't care anymore":         "apathy",
    "lost interest":                "apathy",

    # ── Autonomic Dysfunction ─────────────────────────────────────────────────
    "autonomic dysfunction":        "autonomic dysfunction",
    "autonomic symptoms":           "autonomic dysfunction",
    "dysautonomia":                 "autonomic dysfunction",
    "autonomic problems":           "autonomic dysfunction",
    "constipation":                 "constipation",
    "bowel problems":               "constipation",
    "difficulty passing stool":     "constipation",
    "slow bowel":                   "constipation",
    "excessive sweating":           "excessive sweating",
    "hyperhidrosis":                "excessive sweating",
    "sweating problems":            "excessive sweating",
    "profuse sweating":             "excessive sweating",
    "incontinence":                 "incontinence",
    "bladder control problems":     "incontinence",
    "urinary incontinence":         "incontinence",
    "loss of bladder control":      "incontinence",

    # ── Loss of Smell ─────────────────────────────────────────────────────────
    "olfactory dysfunction":        "loss of smell",
    "loss of smell":                "loss of smell",
    "reduced smell":                "loss of smell",
    "anosmia":                      "loss of smell",
    "hyposmia":                     "loss of smell",
    "can't smell":                  "loss of smell",
    "no sense of smell":            "loss of smell",

    # ── Respiratory ───────────────────────────────────────────────────────────
    "respiratory impairment":       "respiratory difficulty",
    "breathing problems":           "respiratory difficulty",
    "difficulty breathing":         "respiratory difficulty",
    "shortness of breath":          "respiratory difficulty",
    "respiratory failure":          "respiratory difficulty",
    "trouble breathing":            "respiratory difficulty",
    "can't breathe":                "respiratory difficulty",
    "short of breath":              "respiratory difficulty",

    # ── Sleep Disturbance ─────────────────────────────────────────────────────
    "sleep disturbance":            "sleep disturbance",
    "sleep problems":               "sleep disturbance",
    "insomnia":                     "sleep disturbance",
    "poor sleep":                   "sleep disturbance",
    "rem sleep disorder":           "sleep disturbance",
    "sleep disorder":               "sleep disturbance",
    "trouble sleeping":             "sleep disturbance",
    "lose sleep":                   "sleep disturbance",
    "losing sleep":                 "sleep disturbance",
    "can't sleep at night":         "sleep disturbance",
    "can't sleep":                  "sleep disturbance",
    "i lose sleep":                 "sleep disturbance",
    "difficulty sleeping":          "sleep disturbance",
    "difficulties sleeping":        "sleep disturbance",
    "having difficulties sleeping": "sleep disturbance",
    "sleep difficulty":             "sleep disturbance",
    "wake up at night":             "sleep disturbance",
    "restless sleep":               "sleep disturbance",
    "i can't sleep":                "sleep disturbance",
    "having trouble sleeping":      "sleep disturbance",
    "sleep issues":                 "sleep disturbance",
    "disrupted sleep":              "sleep disturbance",
    "waking up at night":           "sleep disturbance",
    "nightmares":                   "sleep disturbance",
    "acting out dreams":            "sleep disturbance",

    # ── Sensory / Paresthesia ─────────────────────────────────────────────────
    "paresthesia":                  "tingling",
    "pins and needles":             "tingling",
    "tingling sensation":           "tingling",
    "abnormal sensation":           "tingling",
    "tingling":                     "tingling",
    "sensory symptoms":             "tingling",
    "sensory changes":              "tingling",
    "sensory disturbance":          "tingling",
    "numbness":                     "numbness",
    "numb":                         "numbness",
    "can't feel":                   "numbness",
    "loss of feeling":              "numbness",
    "burning sensation":            "tingling",
    "electric feeling":             "tingling",

    # ── Pain ──────────────────────────────────────────────────────────────────
    "pain":                         "pain",
    "chronic pain":                 "pain",
    "neuropathic pain":             "pain",
    "aching":                       "pain",
    "muscle pain":                  "pain",

    # ── Weight Loss / Muscle Atrophy ──────────────────────────────────────────
    "weight loss":                  "weight loss",
    "losing weight":                "weight loss",
    "unintentional weight loss":    "weight loss",
    "muscle atrophy":               "muscle atrophy",
    "muscles getting smaller":      "muscle atrophy",
    "muscle loss":                  "muscle atrophy",

    # ── Chest Pain ────────────────────────────────────────────────────────────
    "chest pain":                   "chest pain",
    "my chest hurts":               "chest pain",
    "chest hurts":                  "chest pain",
    "my chest hurt":                "chest pain",

    # ── Vision ────────────────────────────────────────────────────────────────
    "vision problems":              "visual disturbance",
    "blurry vision":                "visual disturbance",
    "double vision":                "visual disturbance",
    "can't see clearly":            "visual disturbance",
    "losing vision":                "visual disturbance",
    "blurred vision":               "visual disturbance",

    # ── Fatigue ───────────────────────────────────────────────────────────────
    "fatigue":                      "fatigue",
    "tired":                        "fatigue",
    "exhausted":                    "fatigue",
    "always tired":                 "fatigue",
    "no energy":                    "fatigue",
    "extreme tiredness":            "fatigue",

    # ── Dizziness ─────────────────────────────────────────────────────────────
    "dizziness":                    "dizziness",
    "dizzy":                        "dizziness",
    "lightheaded":                  "dizziness",
    "room spinning":                "dizziness",
    "vertigo":                      "vertigo",

    # ── Headache ──────────────────────────────────────────────────────────────
    "headache":                     "headache",
    "head pain":                    "headache",
    "migraine":                     "headache",
    "severe headache":              "headache",
    "worst headache of my life":    "headache",

    # ── Seizure ───────────────────────────────────────────────────────────────
    "seizure":                      "seizure",
    "seizures":                     "seizure",
    "fits":                         "seizure",
    "convulsions":                  "seizure",
    "epilepsy":                     "seizure",

    # ── Loss of Consciousness ─────────────────────────────────────────────────
    "blacking out":                 "loss of consciousness",
    "passing out":                  "loss of consciousness",
    "fainting":                     "loss of consciousness",
    "loss of consciousness":        "loss of consciousness",

    # ── Stroke-specific ───────────────────────────────────────────────────────
    "face drooping":                "facial drooping",
    "face droop":                   "facial drooping",
    "facial weakness":              "facial drooping",
    "smile looks uneven":           "facial drooping",
    "facial drooping":              "facial drooping",
    "sudden numbness":              "sudden numbness",
    "sudden weakness":              "sudden weakness",
    "sudden confusion":             "sudden confusion",

    # ── Metabolic ─────────────────────────────────────────────────────────────
    "insulin resistance":           "metabolic dysfunction",
    "blood sugar problems":         "metabolic dysfunction",
    "metabolic dysfunction":        "metabolic dysfunction",
}


# =============================================================================
# PART 2 — REGEX PATTERNS
# Flexible matching for contractions, variable word order, plural forms.
# Each entry: (compiled_regex, canonical_symptom_name)
# ALL matches are collected (not just first).
# =============================================================================

_RAW_PATTERNS = [
    # Memory
    (r"can'?t\s+remember|cannot\s+remember",                        "memory loss"),
    (r"los(?:ing|t)\s+(?:my\s+)?memory",                            "memory loss"),
    (r"forget(?:ting|ful|fulness)?\b",                              "memory loss"),
    (r"(?:short[- ]?term\s+)?memory\s+(?:loss|problem|issue|trouble|impairment)", "memory loss"),
    (r"don'?t\s+remember",                                          "memory loss"),
    (r"can'?t\s+recall",                                            "memory loss"),
    (r"memory\s+(?:is|has\s+been|getting)\s+(?:bad|worse|poor)",    "memory loss"),
    (r"brain\s+fog",                                                "cognitive decline"),
    (r"can'?t\s+(?:think|concentrate|focus)",                       "cognitive decline"),
    (r"getting\s+(?:lost|confused)",                                "confusion"),
    (r"(?:poor|bad|impaired)\s+(?:judgment|reasoning|decisions?)",  "cognitive decline"),

    # Tremor / Shaking
    (r"(?:hands?|arms?|legs?|body|fingers?)\s+(?:is\s+|are\s+)?(?:shak(?:ing|y|es?)|trembl(?:ing|es?))", "tremor"),
    (r"shak(?:ing|y|es?)\b",                                        "tremor"),
    (r"trembl(?:ing|es?|or)\b",                                     "tremor"),
    (r"tremors?\b",                                                  "tremor"),
    (r"uncontrollab\w+\s+(?:shaking|movements?|tremor)",             "tremor"),

    # Rigidity / Stiffness
    (r"(?:muscle|body|joint|neck|back)s?\s+(?:feel(?:ing)?\s+)?(?:stiff|rigid|tight)", "rigidity"),
    (r"stiff(?:ness)?\b",                                           "rigidity"),
    (r"rigid(?:ity)?\b",                                            "rigidity"),

    # Movement slowness
    (r"slow(?:ing|ed)?\s+(?:down|movements?|walking)",              "bradykinesia"),
    (r"moving\s+(?:very\s+|really\s+)?slowly",                     "bradykinesia"),
    (r"tak(?:ing|es?)\s+(?:longer|more\s+time)\s+to\s+(?:move|walk|do)", "bradykinesia"),

    # Gait / Walking
    (r"(?:trouble|difficulty|hard(?:er)?|problem)\s+walk(?:ing)?",  "gait disturbance"),
    (r"can'?t\s+walk\b",                                            "gait disturbance"),
    (r"shuffl(?:ing|es?)\s+(?:when\s+)?(?:walk(?:ing)?|gait)",     "gait disturbance"),

    # Balance / Falls
    (r"los(?:ing|t|es?)\s+(?:my\s+)?balance",                      "balance problems"),
    (r"(?:problem|difficulty|trouble)\s+(?:with\s+)?balance",       "balance problems"),
    (r"keep(?:ing|s?)?\s+fall(?:ing|en|s?)\b",                     "falls"),
    (r"fall(?:ing|s?)?\s+(?:over|down|a\s+lot)",                   "falls"),

    # Weakness
    (r"(?:arm|leg|limb|muscle)s?\s+(?:feel(?:ing)?\s+)?(?:weak|numb|heavy)", "muscle weakness"),
    (r"feel(?:ing|s?)?\s+(?:very\s+|really\s+)?weak\b",            "muscle weakness"),
    (r"can'?t\s+(?:lift|grip|hold|open)",                          "muscle weakness"),

    # Speech
    (r"slurr(?:ing|ed)\s+(?:speech|words?|my\s+words?)",           "dysarthria"),
    (r"(?:trouble|difficult(?:y|ies?)|hard(?:er)?)\s+(?:speak(?:ing)?|talk(?:ing)?)", "dysarthria"),
    (r"can'?t\s+(?:speak|talk)\b",                                  "dysarthria"),
    (r"speech\s+(?:is\s+|has\s+(?:become\s+))?(?:slurred|unclear|nasal|soft|quiet|weak)", "dysarthria"),
    (r"(?:can'?t\s+find|forget(?:ting)?)\s+(?:the\s+)?(?:right\s+)?words?", "aphasia"),
    (r"words?\s+(?:come\s+out\s+wrong|don'?t\s+come\s+out\s+right)", "aphasia"),

    # Swallowing
    (r"(?:trouble|difficulty|hard(?:er)?)\s+swallow(?:ing)?",       "dysphagia"),
    (r"can'?t\s+swallow\b",                                         "dysphagia"),
    (r"chok(?:ing|es?)\s+(?:on\s+food|when\s+eating|when\s+swallowing)", "dysphagia"),

    # Voice
    (r"voice\s+(?:is\s+|has\s+(?:become\s+))?(?:soft|quiet|weak|low|getting\s+quieter)", "hypophonia"),
    (r"speak(?:ing|s?)?\s+(?:very\s+)?softly",                     "hypophonia"),

    # Facial expression
    (r"(?:masked|expressionless|blank)\s+face",                     "hypomimia"),
    (r"face\s+(?:droop(?:ing)?|is\s+dropping)",                    "facial drooping"),

    # Sensory
    (r"numb(?:ness)?\b",                                            "numbness"),
    (r"tingl(?:ing|y)\b|pins\s+and\s+needles",                     "tingling"),
    (r"blurr(?:y|ed)\s+vision|double\s+vision",                    "visual disturbance"),
    (r"can'?t\s+see\s+(?:well|clearly|properly)",                  "visual disturbance"),

    # Neuropsychiatric
    (r"see(?:ing|s?)?\s+things?(?:\s+that\s+aren'?t\s+there)?",   "hallucinations"),
    (r"hear(?:ing|s?)?\s+(?:voices?|things?)",                     "hallucinations"),
    (r"feel(?:ing)?\s+(?:very\s+)?(?:down|low|sad|depressed)",     "depression"),
    (r"(?:mood|personality|behaviour|behavior)\s+chang(?:e|es|ing)", "personality change"),
    (r"act(?:ing|s?)?\s+differently\b",                            "personality change"),
    (r"involuntar(?:y|ily)\s+(?:crying|laughing)",                 "pseudobulbar affect"),
    (r"can'?t\s+(?:sit|stay)\s+still\b",                          "akathisia"),

    # Sleep
    (r"can'?t\s+(?:sleep|fall\s+asleep|stay\s+asleep)",           "sleep disturbance"),
    (r"wak(?:ing|es?)\s+up\s+(?:at\s+night|during\s+the\s+night)", "sleep disturbance"),
    (r"(?:trouble|difficult(?:y|ies?)|problem)\s+sleep(?:ing)?",   "sleep disturbance"),

    # Autonomic
    (r"pass(?:ing|ed)?\s+out|faint(?:ing|ed)?|black(?:ing|ed)?\s+out", "loss of consciousness"),
    (r"short(?:ness)?\s+of\s+breath|trouble\s+breath(?:ing)?|can'?t\s+breath(?:e)?", "respiratory difficulty"),
    (r"los(?:ing|t)\s+weight|weight\s+loss",                       "weight loss"),
    (r"muscle\s+(?:wast(?:ing|ed)|loss|atrophy)",                  "muscle atrophy"),
    (r"(?:face|facial)\s+droop(?:ing)?",                           "facial drooping"),
    (r"loss\s+of\s+(?:smell|sense\s+of\s+smell)|can'?t\s+smell",  "loss of smell"),
    (r"(?:excessive|too\s+much)\s+sweat(?:ing)?|hyperhidrosis",    "excessive sweating"),
    (r"(?:bladder|urinary)\s+(?:control\s+)?(?:problem|issue|incontinence)", "incontinence"),
]

# Pre-compile for performance
SYMPTOM_REGEX_PATTERNS: list[tuple] = [
    (re.compile(pat, re.IGNORECASE), canonical)
    for pat, canonical in _RAW_PATTERNS
]


# =============================================================================
# PART 3 — PUBLIC LOOKUP FUNCTION
# =============================================================================

def lookup_symptoms(user_text: str) -> list[str]:
    """
    Return a deduplicated list of canonical symptom names found in user_text.
    Checks exact phrases first (longer phrases before shorter), then regex.

    Examples
    --------
    >>> lookup_symptoms("I have tremors and rigidity")
    ['tremor', 'rigidity']

    >>> lookup_symptoms("I lose memory and forget things")
    ['memory loss']

    >>> lookup_symptoms("severe, I also have memory problem")
    ['memory loss']
    """
    text_lower = user_text.lower()
    found      = []
    seen       = set()

    # Pass 1 — exact phrases (longer first avoids partial match shadowing)
    sorted_phrases = sorted(SYMPTOM_SYNONYMS.keys(), key=len, reverse=True)
    for phrase in sorted_phrases:
        if phrase in text_lower:
            canonical = SYMPTOM_SYNONYMS[phrase]
            if canonical not in seen:
                found.append(canonical)
                seen.add(canonical)

    # Pass 2 — regex (contractions, variable word order)
    for pattern, canonical in SYMPTOM_REGEX_PATTERNS:
        if canonical not in seen and pattern.search(text_lower):
            found.append(canonical)
            seen.add(canonical)

    return found


# =============================================================================
# PART 4 — SELF-TEST
# =============================================================================

if __name__ == "__main__":
    test_cases = [
        ("i lose memory",                       ["memory loss"]),
        ("i forget things",                     ["memory loss"]),
        ("I'm very forgetful",                  ["memory loss"]),
        ("can't remember anything",             ["memory loss"]),
        ("memory problem",                      ["memory loss"]),
        ("I have tremors and rigidity",         ["tremor", "rigidity"]),
        ("my hands shake",                      ["tremor"]),
        ("I also have memory problem",          ["memory loss"]),
        ("severe, I also have memory problem",  ["memory loss"]),
        ("stiff muscles and losing balance",    ["rigidity", "balance problems"]),
        ("trouble swallowing and soft voice",   ["dysphagia", "hypophonia"]),
        ("brain fog and forgetting recent events", ["cognitive decline", "memory loss"]),
        ("hello how are you",                   []),
    ]

    print("Synonym lookup self-test")
    print("=" * 55)
    passed = 0
    for text, expected in test_cases:
        result = lookup_symptoms(text)
        ok     = all(e in result for e in expected)
        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        print(f"  {status}  '{text}'")
        if not ok:
            print(f"       expected: {expected}")
            print(f"       got:      {result}")

    print(f"\n{passed}/{len(test_cases)} tests passed")
