"""
phase_eval.py — Standalone Evaluation Module
=============================================
Runs four independent evaluations WITHOUT modifying any existing files:

  Section 1 — NER Evaluation
      Measures precision, recall and F1 for scispaCy entity extraction
      against a small hand-labelled gold standard of 10 abstracts.

  Section 2 — RAG Retrieval Evaluation
      Measures retrieval quality (Hit Rate @ k) and answer quality
      (ROUGE-1, ROUGE-2, ROUGE-L) for the PICOS RAG pipeline.

  Section 3 — ROUGE Chatbot Evaluation
      Evaluates the full chatbot responses on a fixed Q&A benchmark
      using ROUGE-1, ROUGE-2, and ROUGE-L scores.

  Section 4 — Error Analysis
      Four complementary error analyses for the chatbot:
        4a  Classifier confusion matrix — which diseases are confused with which
        4b  RAG hedging detector — answers where the LLM said "not enough info"
        4c  PICOS field completeness — how many abstracts have each field filled
        4d  Low-ROUGE answer audit — worst-scoring answers inspected for root cause

What is ROUGE?
--------------
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures how
much overlap exists between a generated text and a reference (gold) text.

  ROUGE-N   — overlapping n-gram counts (N=1 unigrams, N=2 bigrams)
  ROUGE-L   — longest common subsequence (word-level)

For each variant, three scores are computed:
  Precision = overlapping n-grams / total n-grams in GENERATED text
  Recall    = overlapping n-grams / total n-grams in REFERENCE text
  F1        = 2 * P * R / (P + R)

Higher is better (range 0–1).  In NLP literature, ROUGE-1 F1 > 0.40 and
ROUGE-L F1 > 0.35 are generally considered acceptable for summarisation /
QA tasks.  For domain-specific medical QA, scores may be lower because
correct answers can be phrased in many valid ways.

Install:
    pip install rouge-score --break-system-packages

Run:
    python phase_eval.py                  # all sections
    python phase_eval.py --section ner    # NER only
    python phase_eval.py --section rag    # RAG retrieval + ROUGE only
    python phase_eval.py --section rouge  # ROUGE chatbot only
    python phase_eval.py --section error  # Error analysis only
"""

import argparse
import json
import sqlite3
import os
import sys

DB_PATH = "abstracts.db"

# ═══════════════════════════════════════════════════════════════════════════════
# GOLD STANDARD DATA — hand-labelled benchmark sets
# These are fixed reference sets used for evaluation only.
# ═══════════════════════════════════════════════════════════════════════════════

# ── NER Gold Standard ─────────────────────────────────────────────────────────
# Each entry: abstract snippet + expected entity labels
# (disease / chemical) that scispaCy should detect.
# Sourced from well-known neurodegenerative disease facts.

NER_GOLD = [
    {
        "text": "Levodopa combined with carbidopa is the gold standard treatment for Parkinson's disease.",
        "expected": [
            {"text": "Levodopa",           "label": "CHEMICAL"},
            {"text": "carbidopa",          "label": "CHEMICAL"},
            {"text": "Parkinson's disease","label": "DISEASE"},
        ]
    },
    {
        "text": "Donepezil and memantine are approved for the treatment of Alzheimer's disease.",
        "expected": [
            {"text": "Donepezil",          "label": "CHEMICAL"},
            {"text": "memantine",          "label": "CHEMICAL"},
            {"text": "Alzheimer's disease","label": "DISEASE"},
        ]
    },
    {
        "text": "Riluzole is the only FDA-approved drug for amyotrophic lateral sclerosis.",
        "expected": [
            {"text": "Riluzole",                      "label": "CHEMICAL"},
            {"text": "amyotrophic lateral sclerosis", "label": "DISEASE"},
        ]
    },
    {
        "text": "Patients with Parkinson's disease often experience resting tremor and bradykinesia.",
        "expected": [
            {"text": "Parkinson's disease", "label": "DISEASE"},
        ]
    },
    {
        "text": "Beta-amyloid plaques and tau neurofibrillary tangles are hallmarks of Alzheimer's disease.",
        "expected": [
            {"text": "Beta-amyloid",        "label": "CHEMICAL"},
            {"text": "tau",                 "label": "CHEMICAL"},
            {"text": "Alzheimer's disease", "label": "DISEASE"},
        ]
    },
    {
        "text": "Pramipexole and ropinirole are dopamine agonists used in Parkinson's disease management.",
        "expected": [
            {"text": "Pramipexole",         "label": "CHEMICAL"},
            {"text": "ropinirole",          "label": "CHEMICAL"},
            {"text": "dopamine",            "label": "CHEMICAL"},
            {"text": "Parkinson's disease", "label": "DISEASE"},
        ]
    },
    {
        "text": "Edaravone has been approved for ALS treatment to slow functional decline.",
        "expected": [
            {"text": "Edaravone", "label": "CHEMICAL"},
            {"text": "ALS",       "label": "DISEASE"},
        ]
    },
    {
        "text": "Cognitive impairment and memory loss are early symptoms of Alzheimer's disease.",
        "expected": [
            {"text": "Alzheimer's disease", "label": "DISEASE"},
        ]
    },
    {
        "text": "Deep brain stimulation targeting the subthalamic nucleus is effective for Parkinson's disease tremor.",
        "expected": [
            {"text": "Parkinson's disease", "label": "DISEASE"},
        ]
    },
    {
        "text": "Motor neuron degeneration in ALS leads to progressive paralysis and respiratory failure.",
        "expected": [
            {"text": "ALS", "label": "DISEASE"},
        ]
    },
    # ── Huntington's Disease ──
    {
        "text": "Tetrabenazine is approved for chorea associated with Huntington's disease.",
        "expected": [
            {"text": "Tetrabenazine",       "label": "CHEMICAL"},
            {"text": "chorea",              "label": "DISEASE"},
            {"text": "Huntington's disease","label": "DISEASE"},
        ]
    },
    {
        "text": "CAG repeat expansion in the HTT gene causes Huntington's disease through mutant huntingtin protein.",
        "expected": [
            {"text": "Huntington's disease", "label": "DISEASE"},
        ]
    },
    # ── Dementia ──
    {
        "text": "Vascular dementia and Lewy body dementia are common subtypes of dementia in older adults.",
        "expected": [
            {"text": "Vascular dementia",   "label": "DISEASE"},
            {"text": "Lewy body dementia",  "label": "DISEASE"},
            {"text": "dementia",            "label": "DISEASE"},
        ]
    },
    {
        "text": "Rivastigmine and galantamine are cholinesterase inhibitors used to manage dementia symptoms.",
        "expected": [
            {"text": "Rivastigmine",  "label": "CHEMICAL"},
            {"text": "galantamine",   "label": "CHEMICAL"},
            {"text": "dementia",      "label": "DISEASE"},
        ]
    },
    # ── Stroke ──
    {
        "text": "Tissue plasminogen activator is administered intravenously to treat acute ischaemic stroke.",
        "expected": [
            {"text": "Tissue plasminogen activator", "label": "CHEMICAL"},
            {"text": "ischaemic stroke",             "label": "DISEASE"},
        ]
    },
    {
        "text": "Aspirin and clopidogrel reduce the risk of recurrent stroke in patients with transient ischaemic attack.",
        "expected": [
            {"text": "Aspirin",                    "label": "CHEMICAL"},
            {"text": "clopidogrel",                "label": "CHEMICAL"},
            {"text": "stroke",                     "label": "DISEASE"},
            {"text": "transient ischaemic attack", "label": "DISEASE"},
        ]
    },
]


# ── RAG + ROUGE Gold Standard Q&A ─────────────────────────────────────────────
# Fixed benchmark questions with reference answers.
# Reference answers are concise summaries of well-established clinical facts.
# The RAG system's generated answers are scored against these.

ROUGE_QA_BENCHMARK = [
    # ── Parkinson's Disease ──
    {
        "question": "What interventions have been studied for Parkinson's disease tremor?",
        "reference": (
            "Levodopa and dopamine agonists such as pramipexole and ropinirole are the "
            "primary pharmacological treatments for Parkinson's disease tremor. "
            "Deep brain stimulation of the subthalamic nucleus is an effective surgical "
            "intervention for patients with medication-resistant tremor."
        ),
        "disease": "Parkinson's Disease"
    },
    {
        "question": "What are the outcomes of levodopa therapy in Parkinson's disease?",
        "reference": (
            "Levodopa therapy significantly reduces motor symptoms including resting tremor, "
            "rigidity, and bradykinesia in Parkinson's disease patients. "
            "Long-term use is associated with motor fluctuations and dyskinesia. "
            "UPDRS scores typically show improvement with levodopa treatment."
        ),
        "disease": "Parkinson's Disease"
    },
    # ── Alzheimer's Disease ──
    {
        "question": "What treatments have been studied for cognitive decline in Alzheimer's disease?",
        "reference": (
            "Cholinesterase inhibitors including donepezil, rivastigmine, and galantamine "
            "are used to treat mild to moderate Alzheimer's disease. "
            "Memantine, an NMDA receptor antagonist, is approved for moderate to severe "
            "Alzheimer's disease. These drugs slow cognitive decline but do not halt progression."
        ),
        "disease": "Alzheimer's Disease"
    },
    {
        "question": "What are the risk factors for Alzheimer's disease?",
        "reference": (
            "Age is the strongest risk factor for Alzheimer's disease. "
            "The APOE epsilon 4 allele significantly increases genetic risk. "
            "Cardiovascular risk factors including hypertension, diabetes, and obesity "
            "are associated with increased Alzheimer's risk. "
            "Family history and traumatic brain injury are also established risk factors."
        ),
        "disease": "Alzheimer's Disease"
    },
    # ── ALS and Huntington's Disease ──
    {
        "question": "What is the prognosis for patients with ALS?",
        "reference": (
            "Amyotrophic lateral sclerosis has a poor prognosis with median survival of "
            "two to five years from symptom onset. Respiratory failure is the most common "
            "cause of death. Riluzole extends survival by approximately three months. "
            "A small proportion of patients survive more than ten years."
        ),
        "disease": "ALS and Huntington's Disease"
    },
    {
        "question": "What are the symptoms and treatments for Huntington's disease?",
        "reference": (
            "Huntington's disease causes progressive motor dysfunction including chorea, "
            "cognitive decline, and psychiatric symptoms. "
            "Tetrabenazine and deutetrabenazine are approved to reduce chorea. "
            "There is currently no disease-modifying treatment. "
            "Symptoms typically begin between ages 30 and 50."
        ),
        "disease": "ALS and Huntington's Disease"
    },
    # ── Dementia and Mild Cognitive Impairment ──
    {
        "question": "What interventions are used for vascular dementia and Lewy body dementia?",
        "reference": (
            "Vascular dementia management focuses on controlling cardiovascular risk factors "
            "including hypertension, diabetes, and antiplatelet therapy. "
            "Lewy body dementia is treated with cholinesterase inhibitors such as rivastigmine. "
            "Antipsychotics must be avoided in Lewy body dementia due to severe sensitivity reactions."
        ),
        "disease": "Dementia and Mild Cognitive Impairment"
    },
    # ── Stroke ──
    {
        "question": "What are the acute treatments for ischaemic stroke?",
        "reference": (
            "Intravenous tissue plasminogen activator administered within 4.5 hours of onset "
            "is the primary thrombolytic treatment for acute ischaemic stroke. "
            "Mechanical thrombectomy is recommended for large vessel occlusion. "
            "Aspirin is given for secondary prevention to reduce recurrent stroke risk."
        ),
        "disease": "Stroke"
    },
]


# ── RAG Retrieval Gold Standard ────────────────────────────────────────────────
# For hit-rate evaluation: we check whether retrieved abstracts contain
# at least one abstract from the expected disease category.

RAG_RETRIEVAL_BENCHMARK = [
    # Parkinson's Disease
    {"query": "levodopa treatment motor symptoms",           "expected_disease": "Parkinson's Disease"},
    {"query": "deep brain stimulation tremor subthalamic",   "expected_disease": "Parkinson's Disease"},
    {"query": "dopamine agonist pramipexole bradykinesia",   "expected_disease": "Parkinson's Disease"},
    # Alzheimer's Disease
    {"query": "donepezil cognitive improvement randomised",  "expected_disease": "Alzheimer's Disease"},
    {"query": "beta amyloid plaque tau pathology",           "expected_disease": "Alzheimer's Disease"},
    {"query": "APOE epsilon genetic risk memory loss",       "expected_disease": "Alzheimer's Disease"},
    # ALS and Huntington's Disease
    {"query": "riluzole survival amyotrophic lateral",       "expected_disease": "ALS and Huntington's Disease"},
    {"query": "respiratory failure muscle weakness ALS",     "expected_disease": "ALS and Huntington's Disease"},
    {"query": "CAG repeat huntingtin chorea motor decline",  "expected_disease": "ALS and Huntington's Disease"},
    # Dementia and Mild Cognitive Impairment
    {"query": "vascular dementia cognitive decline hypertension",       "expected_disease": "Dementia and Mild Cognitive Impairment"},
    {"query": "Lewy body dementia rivastigmine hallucinations",         "expected_disease": "Dementia and Mild Cognitive Impairment"},
    {"query": "mild cognitive impairment progression risk conversion",  "expected_disease": "Dementia and Mild Cognitive Impairment"},
    # Stroke
    {"query": "tissue plasminogen activator thrombolysis acute stroke", "expected_disease": "Stroke"},
    {"query": "mechanical thrombectomy large vessel occlusion",         "expected_disease": "Stroke"},
    {"query": "aspirin antiplatelet secondary prevention stroke",       "expected_disease": "Stroke"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — NER EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_ner():
    """
    Runs scispaCy NER on the gold-standard snippets and computes
    token-level precision, recall, and F1 for DISEASE and CHEMICAL entities.

    Matching strategy: exact text match (case-insensitive).
    """
    print("\n" + "═" * 60)
    print("SECTION 1 — NER EVALUATION (scispaCy vs Gold Standard)")
    print("═" * 60)

    try:
        import spacy
        nlp = spacy.load("en_ner_bc5cdr_md")
    except OSError:
        print("[SKIP] en_ner_bc5cdr_md not found. Run phase2_ner_picos.py install steps first.")
        return

    true_positives  = 0
    false_positives = 0
    false_negatives = 0

    label_stats = {"DISEASE": {"tp": 0, "fp": 0, "fn": 0},
                   "CHEMICAL": {"tp": 0, "fp": 0, "fn": 0}}

    for i, sample in enumerate(NER_GOLD):
        doc = nlp(sample["text"])

        predicted = {(ent.text.lower(), ent.label_) for ent in doc.ents}
        expected  = {(e["text"].lower(), e["label"]) for e in sample["expected"]}

        tp = predicted & expected
        fp = predicted - expected
        fn = expected  - predicted

        true_positives  += len(tp)
        false_positives += len(fp)
        false_negatives += len(fn)

        # Per-label breakdown
        for (text, label) in tp:
            if label in label_stats:
                label_stats[label]["tp"] += 1
        for (text, label) in fp:
            if label in label_stats:
                label_stats[label]["fp"] += 1
        for (text, label) in fn:
            if label in label_stats:
                label_stats[label]["fn"] += 1

        if fp or fn:
            print(f"\n  Sample {i+1}: '{sample['text'][:60]}...'")
            if fp: print(f"    False positives (over-predicted): {fp}")
            if fn: print(f"    False negatives (missed):          {fn}")

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    print("\n── Overall NER Results ──")
    p, r, f = prf(true_positives, false_positives, false_negatives)
    print(f"  Precision : {p:.3f}")
    print(f"  Recall    : {r:.3f}")
    print(f"  F1 Score  : {f:.3f}")

    print("\n── Per-Label Breakdown ──")
    for label, s in label_stats.items():
        lp, lr, lf = prf(s["tp"], s["fp"], s["fn"])
        print(f"  {label:<10}  P={lp:.3f}  R={lr:.3f}  F1={lf:.3f}  "
              f"(TP={s['tp']} FP={s['fp']} FN={s['fn']})")

    print(f"\n  Gold standard size : {len(NER_GOLD)} sentences")
    print(f"  Total expected     : {sum(len(s['expected']) for s in NER_GOLD)} entities")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RAG RETRIEVAL EVALUATION (Hit Rate @ k)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_rag_retrieval():
    """
    Tests whether the FAISS retriever returns at least one abstract from the
    expected disease category in the top-k results (Hit Rate @ k).

    Hit Rate @ k = (queries with ≥1 correct disease in top k) / total queries
    """
    print("\n" + "═" * 60)
    print("SECTION 2 — RAG RETRIEVAL EVALUATION (Hit Rate @ k)")
    print("═" * 60)

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from rag_pipeline import PICOSRetriever
        retriever = PICOSRetriever()
    except Exception as e:
        print(f"[SKIP] Could not load PICOSRetriever: {e}")
        print("       Run phase4_rag.py first to build the FAISS index.")
        return

    k_values = [1, 3, 5]
    hits = {k: 0 for k in k_values}

    print(f"\n  Running {len(RAG_RETRIEVAL_BENCHMARK)} retrieval queries...\n")

    for item in RAG_RETRIEVAL_BENCHMARK:
        query    = item["query"]
        expected = item["expected_disease"]

        # Retrieve at the largest k then check subsets
        results = retriever.retrieve(query, k=max(k_values))
        retrieved_diseases = [r["disease"] for r in results]

        for k in k_values:
            top_k = retrieved_diseases[:k]
            if expected in top_k:
                hits[k] += 1

        match_icon = "✓" if expected in retrieved_diseases[:5] else "✗"
        top1 = retrieved_diseases[0] if retrieved_diseases else "none"
        print(f"  {match_icon} '{query[:45]:<45}' | expected={expected:<12} | top1={top1}")

    total = len(RAG_RETRIEVAL_BENCHMARK)
    print("\n── Hit Rate Results ──")
    for k in k_values:
        hr = hits[k] / total
        print(f"  Hit Rate @ {k}  : {hits[k]}/{total} = {hr:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ROUGE CHATBOT EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rouge(hypothesis: str, reference: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores between a generated
    hypothesis string and a reference string.

    Uses the `rouge-score` library (pip install rouge-score).

    Returns dict with keys: rouge1, rouge2, rougeL
    Each value is a dict with: precision, recall, fmeasure
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("[ERROR] rouge-score not installed. Run:")
        print("        pip install rouge-score --break-system-packages")
        return {}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)

    return {
        "rouge1": {
            "precision": round(scores["rouge1"].precision,  4),
            "recall":    round(scores["rouge1"].recall,     4),
            "fmeasure":  round(scores["rouge1"].fmeasure,   4),
        },
        "rouge2": {
            "precision": round(scores["rouge2"].precision,  4),
            "recall":    round(scores["rouge2"].recall,     4),
            "fmeasure":  round(scores["rouge2"].fmeasure,   4),
        },
        "rougeL": {
            "precision": round(scores["rougeL"].precision,  4),
            "recall":    round(scores["rougeL"].recall,     4),
            "fmeasure":  round(scores["rougeL"].fmeasure,   4),
        },
    }


def evaluate_rouge():
    """
    Runs the full RAG pipeline on each benchmark question, then scores
    the generated answer against the reference using ROUGE-1, ROUGE-2,
    and ROUGE-L.

    Interpretation guide (printed at the end):
      ROUGE-1 F1 > 0.40  — good unigram overlap
      ROUGE-2 F1 > 0.15  — good bigram overlap (harder to achieve)
      ROUGE-L F1 > 0.35  — good longest-sequence overlap
    """
    print("\n" + "═" * 60)
    print("SECTION 3 — ROUGE CHATBOT EVALUATION")
    print("═" * 60)

    # Check rouge-score is available
    try:
        from rouge_score import rouge_scorer  # noqa: F401
    except ImportError:
        print("[ERROR] rouge-score not installed.")
        print("        pip install rouge-score --break-system-packages")
        return

    # Load RAG pipeline
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from rag_pipeline import PICOSRetriever, RAGAnswerGenerator
        retriever = PICOSRetriever()
        rag       = RAGAnswerGenerator(retriever)
    except Exception as e:
        print(f"[SKIP] Could not load RAG pipeline: {e}")
        print("       Run phase4_rag.py first to build the FAISS index.")
        return

    all_r1, all_r2, all_rl = [], [], []

    print(f"\n  Evaluating {len(ROUGE_QA_BENCHMARK)} questions...\n")
    print(f"  {'Question':<50} {'R-1':>6} {'R-2':>6} {'R-L':>6}")
    print(f"  {'-'*50} {'------':>6} {'------':>6} {'------':>6}")

    for item in ROUGE_QA_BENCHMARK:
        question  = item["question"]
        reference = item["reference"]
        disease   = item.get("disease")

        result    = rag.answer(question, k=5, filter_disease=disease)
        generated = result.get("answer", "")

        scores = compute_rouge(generated, reference)
        if not scores:
            continue

        r1 = scores["rouge1"]["fmeasure"]
        r2 = scores["rouge2"]["fmeasure"]
        rl = scores["rougeL"]["fmeasure"]

        all_r1.append(r1)
        all_r2.append(r2)
        all_rl.append(rl)

        print(f"  {question[:50]:<50} {r1:>6.3f} {r2:>6.3f} {rl:>6.3f}")

    if not all_r1:
        print("  No scores computed.")
        return

    avg_r1 = sum(all_r1) / len(all_r1)
    avg_r2 = sum(all_r2) / len(all_r2)
    avg_rl = sum(all_rl) / len(all_rl)

    print(f"\n── Average ROUGE Scores (n={len(all_r1)}) ──")
    print(f"  ROUGE-1 F1 : {avg_r1:.4f}  (unigram overlap)")
    print(f"  ROUGE-2 F1 : {avg_r2:.4f}  (bigram overlap)")
    print(f"  ROUGE-L F1 : {avg_rl:.4f}  (longest common subsequence)")

    print("\n── Interpretation ──")
    r1_status = "✓ GOOD" if avg_r1 >= 0.40 else ("~ MODERATE" if avg_r1 >= 0.25 else "✗ LOW")
    r2_status = "✓ GOOD" if avg_r2 >= 0.15 else ("~ MODERATE" if avg_r2 >= 0.08 else "✗ LOW")
    rl_status = "✓ GOOD" if avg_rl >= 0.35 else ("~ MODERATE" if avg_rl >= 0.20 else "✗ LOW")
    print(f"  ROUGE-1 : {r1_status}")
    print(f"  ROUGE-2 : {r2_status}")
    print(f"  ROUGE-L : {rl_status}")

    print("""
  Note: In medical QA, ROUGE scores are typically lower than in
  news summarisation because correct answers can be phrased in
  many valid ways. A ROUGE-1 F1 of 0.30–0.45 is reasonable for
  a domain-specific literature-based QA system.
  Lower ROUGE + correct factual content > high ROUGE + wrong facts.
""")

    # ── Detailed per-question breakdown ──
    print("── Detailed Scores ──")
    for i, (item, r1, r2, rl) in enumerate(zip(ROUGE_QA_BENCHMARK, all_r1, all_r2, all_rl)):
        print(f"\n  Q{i+1}: {item['question']}")
        print(f"       ROUGE-1={r1:.4f}  ROUGE-2={r2:.4f}  ROUGE-L={rl:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 4a: Phrases that indicate the LLM could not answer confidently ────────────
HEDGE_PHRASES = [
    "do not contain",
    "does not contain",
    "not enough information",
    "insufficient information",
    "cannot answer",
    "not reported",
    "no relevant",
    "unable to find",
    "not mentioned",
    "not discussed",
    "not addressed",
    "no information",
    "no studies",
    "no abstracts",
]


def _classify_hedge_cause(answer: str, retrieved: list) -> str:
    """
    Returns a short root-cause label for a hedging answer:
      - 'empty_retrieval'   — no abstracts were returned at all
      - 'wrong_disease'     — retrieved abstracts are all for a different disease
      - 'picos_gap'         — abstracts exist but PICOS fields are mostly empty
      - 'query_too_vague'   — catch-all
    """
    if not retrieved:
        return "empty_retrieval"

    diseases = [r.get("disease", "") for r in retrieved]
    if len(set(diseases)) == 1 and diseases[0] not in answer:
        return "wrong_disease"

    empty_picos = sum(
        1 for r in retrieved
        if r.get("I", "not reported") in ("not reported", "", None)
        and r.get("O", "not reported") in ("not reported", "", None)
    )
    if empty_picos >= len(retrieved) // 2:
        return "picos_gap"

    return "query_too_vague"


def _error_4a_classifier_confusion():
    """
    Loads the trained SVM classifier and TF-IDF vectorizer, runs them on the
    held-out 20% test split, and prints a confusion matrix with per-cell counts
    plus the top-5 most-confused abstract titles per error pair.
    """
    print("\n── 4a: Disease Classifier Confusion Matrix ──")
    try:
        import joblib
        clf        = joblib.load("disease_classifier.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        le         = joblib.load("label_encoder.pkl")
    except FileNotFoundError as e:
        print(f"  [SKIP] {e} — run phase3_ml.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT abstract, disease_label, title FROM abstracts WHERE abstract IS NOT NULL"
    ).fetchall()
    conn.close()

    if not rows:
        print("  [SKIP] No abstracts found in database.")
        return

    texts   = [r[0] for r in rows]
    labels  = [r[1] for r in rows]
    titles  = [r[2] for r in rows]

    # Reproduce the same 80/20 stratified split used in training
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    y = le.transform(labels)
    _, X_test_texts, _, y_test, _, test_titles = train_test_split(
        texts, y, titles, test_size=0.2, random_state=42, stratify=y
    )

    X_test = vectorizer.transform(X_test_texts)
    y_pred = clf.predict(X_test)

    classes = le.classes_
    n       = len(classes)

    # Build confusion matrix manually (no sklearn dependency for display)
    matrix = [[0] * n for _ in range(n)]
    errors = {}   # (true_label, pred_label) -> list of titles

    for true_idx, pred_idx, title in zip(y_test, y_pred, test_titles):
        matrix[true_idx][pred_idx] += 1
        if true_idx != pred_idx:
            key = (classes[true_idx], classes[pred_idx])
            errors.setdefault(key, []).append(title)

    # Print matrix
    col_w = 12
    print(f"\n  {'True \\ Pred':<14}", end="")
    for c in classes:
        print(f"{c:>{col_w}}", end="")
    print()
    print("  " + "-" * (14 + col_w * n))

    total_correct = sum(matrix[i][i] for i in range(n))
    total         = sum(matrix[i][j] for i in range(n) for j in range(n))

    for i, true_class in enumerate(classes):
        print(f"  {true_class:<14}", end="")
        for j in range(n):
            cell = matrix[i][j]
            tag  = f"[{cell}]" if i == j else str(cell)
            print(f"{tag:>{col_w}}", end="")
        print()

    accuracy = total_correct / total if total else 0
    print(f"\n  Overall accuracy : {total_correct}/{total} = {accuracy:.3f}")

    # Print most common error pairs
    sorted_errors = sorted(errors.items(), key=lambda x: len(x[1]), reverse=True)
    if sorted_errors:
        print("\n  Most common misclassifications:")
        for (true_c, pred_c), example_titles in sorted_errors[:3]:
            print(f"\n    True={true_c} → Predicted={pred_c}  ({len(example_titles)} cases)")
            for t in example_titles[:3]:
                print(f"      • {(t or 'untitled')[:80]}")
    else:
        print("\n  No misclassifications found — perfect test accuracy.")


def _error_4b_rag_hedging():
    """
    Runs the full RAG pipeline on the ROUGE benchmark questions, detects
    answers that hedge (LLM said it couldn't answer), and diagnoses
    the most likely root cause for each hedge.
    """
    print("\n── 4b: RAG Hedging / Fallback Detection ──")
    try:
        from rag_pipeline import PICOSRetriever, RAGAnswerGenerator
        retriever = PICOSRetriever()
        rag       = RAGAnswerGenerator(retriever)
    except Exception as e:
        print(f"  [SKIP] Could not load RAG pipeline: {e}")
        return

    hedge_count = 0
    root_causes = {}

    for item in ROUGE_QA_BENCHMARK:
        result    = rag.answer(item["question"], k=5, filter_disease=item.get("disease"))
        answer    = result.get("answer", "").lower()
        retrieved = result.get("picos_summary", [])

        hedged = any(phrase in answer for phrase in HEDGE_PHRASES)

        if hedged:
            hedge_count += 1
            cause = _classify_hedge_cause(answer, retrieved)
            root_causes[cause] = root_causes.get(cause, 0) + 1
            print(f"\n  ⚠ HEDGE detected")
            print(f"    Question  : {item['question']}")
            print(f"    Root cause: {cause}")
            print(f"    Answer    : {result.get('answer','')[:200]}...")
        else:
            print(f"  ✓ {item['question'][:65]}")

    total = len(ROUGE_QA_BENCHMARK)
    hedge_rate = hedge_count / total if total else 0
    print(f"\n  Hedge rate : {hedge_count}/{total} = {hedge_rate:.2%}")

    if root_causes:
        print("\n  Root cause breakdown:")
        for cause, count in sorted(root_causes.items(), key=lambda x: -x[1]):
            explanation = {
                "empty_retrieval": "FAISS returned no abstracts for this query",
                "wrong_disease":   "Retrieved abstracts were for the wrong disease",
                "picos_gap":       "Abstracts exist but PICOS fields are mostly empty",
                "query_too_vague": "Query is too broad / no clear PICOS match",
            }.get(cause, cause)
            print(f"    {cause:<20} {count}x  — {explanation}")

    if hedge_count == 0:
        print("  All questions answered confidently — no hedging detected.")


def _error_4c_picos_completeness():
    """
    Queries the database to measure what percentage of abstracts have each
    PICOS field filled in (not NULL and not 'not reported').
    Highlights which fields are sparsely populated — these are the ones
    most likely to cause RAG retrieval gaps.
    """
    print("\n── 4c: PICOS Field Completeness ──")
    if not os.path.exists(DB_PATH):
        print(f"  [SKIP] {DB_PATH} not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    total = c.execute("SELECT COUNT(*) FROM abstracts").fetchone()[0]
    if total == 0:
        print("  [SKIP] Database is empty.")
        conn.close()
        return

    fields = {
        "P — Population":    "picos_population",
        "I — Intervention":  "picos_intervention",
        "C — Comparison":    "picos_comparison",
        "O — Outcome":       "picos_outcome",
        "S — Study design":  "picos_study_design",
    }

    print(f"\n  Total abstracts in DB: {total}\n")
    print(f"  {'Field':<22} {'Filled':>8} {'Empty':>8} {'Coverage':>10}  {'Quality check'}")
    print("  " + "-" * 70)

    for label, col in fields.items():
        filled = c.execute(
            f"SELECT COUNT(*) FROM abstracts "
            f"WHERE {col} IS NOT NULL AND {col} != '' AND {col} != 'not reported'"
        ).fetchone()[0]
        empty    = total - filled
        coverage = filled / total if total else 0

        if coverage >= 0.80:
            quality = "✓ Good"
        elif coverage >= 0.50:
            quality = "~ Moderate — may cause retrieval gaps"
        else:
            quality = "✗ Low — recommend re-running PICOS extraction"

        print(f"  {label:<22} {filled:>8} {empty:>8} {coverage:>9.1%}  {quality}")

    # Per-disease breakdown
    print("\n  Per-disease PICOS coverage (Intervention field):")
    for disease in [
        "Alzheimer's Disease",
        "Parkinson's Disease",
        "ALS and Huntington's Disease",
        "Dementia and Mild Cognitive Impairment",
        "Stroke",
    ]:
        total_d = c.execute(
            "SELECT COUNT(*) FROM abstracts WHERE disease_label=?", (disease,)
        ).fetchone()[0]
        filled_d = c.execute(
            "SELECT COUNT(*) FROM abstracts WHERE disease_label=? "
            "AND picos_intervention IS NOT NULL AND picos_intervention != '' "
            "AND picos_intervention != 'not reported'", (disease,)
        ).fetchone()[0]
        cov = filled_d / total_d if total_d else 0
        print(f"    {disease:<12} {filled_d}/{total_d} = {cov:.1%}")

    conn.close()


def _error_4d_low_rouge_audit():
    """
    Re-runs ROUGE scoring on the benchmark Q&A set, ranks questions by
    ROUGE-1 F1, and for the bottom-2 scorers prints the full generated
    answer and reference so you can inspect why the score was low.

    Root-cause categories:
      - Paraphrase gap   : correct facts but different wording → low ROUGE, good answer
      - Vague answer     : answer is too short / generic → low ROUGE, poor answer
      - Off-topic answer : answer addresses wrong aspect → low ROUGE, poor answer
      - Hallucination    : answer contains unsupported claims
    """
    print("\n── 4d: Low-ROUGE Answer Audit ──")
    try:
        from rouge_score import rouge_scorer as rs
    except ImportError:
        print("  [SKIP] pip install rouge-score --break-system-packages")
        return

    try:
        from rag_pipeline import PICOSRetriever, RAGAnswerGenerator
        retriever = PICOSRetriever()
        rag       = RAGAnswerGenerator(retriever)
    except Exception as e:
        print(f"  [SKIP] Could not load RAG pipeline: {e}")
        return

    scorer  = rs.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    results = []

    for item in ROUGE_QA_BENCHMARK:
        result    = rag.answer(item["question"], k=5, filter_disease=item.get("disease"))
        generated = result.get("answer", "")
        reference = item["reference"]
        scores    = scorer.score(reference, generated)

        r1 = scores["rouge1"].fmeasure
        rl = scores["rougeL"].fmeasure

        # Diagnose root cause
        gen_lower = generated.lower()
        hedged    = any(p in gen_lower for p in HEDGE_PHRASES)
        too_short = len(generated.split()) < 30

        if hedged:
            cause = "Vague / hedging answer — LLM lacked retrieval context"
        elif too_short:
            cause = "Answer too short — insufficient detail generated"
        elif r1 >= 0.35:
            cause = "Paraphrase gap — correct content, different wording (acceptable)"
        else:
            cause = "Off-topic or hallucinated — review retrieved abstracts"

        results.append({
            "question":  item["question"],
            "generated": generated,
            "reference": reference,
            "rouge1":    r1,
            "rougeL":    rl,
            "cause":     cause,
        })

    # Sort ascending by ROUGE-1 (worst first)
    results.sort(key=lambda x: x["rouge1"])

    print(f"\n  All questions ranked by ROUGE-1 (worst → best):\n")
    for i, r in enumerate(results):
        flag = "⚠" if r["rouge1"] < 0.30 else ("~" if r["rouge1"] < 0.40 else "✓")
        print(f"  {flag} [{i+1}] R1={r['rouge1']:.3f}  RL={r['rougeL']:.3f}  "
              f"'{r['question'][:55]}'")

    # Deep-dive on the 2 worst
    audit_n = min(2, len(results))
    print(f"\n  ── Deep-dive: {audit_n} lowest-scoring answers ──")

    for r in results[:audit_n]:
        print(f"\n  Question  : {r['question']}")
        print(f"  ROUGE-1   : {r['rouge1']:.4f}    ROUGE-L: {r['rougeL']:.4f}")
        print(f"  Root cause: {r['cause']}")
        print(f"\n  Reference answer:")
        print(f"    {r['reference']}")
        print(f"\n  Generated answer:")
        for line in r["generated"].split("\n"):
            print(f"    {line}")

    print("""
  Interpreting low ROUGE scores:
    Paraphrase gap   → acceptable; the LLM rewrote the answer in its own words
    Vague/hedging    → fix by enriching PICOS fields or widening retrieval k
    Too short        → increase max_tokens in RAGAnswerGenerator.answer()
    Off-topic        → check that filter_disease is set correctly for the query
    Hallucination    → verify retrieved abstracts actually support the claim
""")


def evaluate_error_analysis():
    """Runs all four error analysis sub-sections."""
    print("\n" + "═" * 60)
    print("SECTION 4 — ERROR ANALYSIS")
    print("═" * 60)
    _error_4a_classifier_confusion()
    _error_4b_rag_hedging()
    _error_4c_picos_completeness()
    _error_4d_low_rouge_audit()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Standalone evaluation — NER, RAG retrieval, and ROUGE chatbot scoring."
    )
    parser.add_argument(
        "--section",
        choices=["ner", "rag", "rouge", "error", "all"],
        default="all",
        help="Which section to run (default: all)"
    )
    args = parser.parse_args()

    # Change to script directory so relative paths (abstracts.db, etc.) resolve
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if args.section in ("ner", "all"):
        evaluate_ner()

    if args.section in ("rag", "all"):
        evaluate_rag_retrieval()

    if args.section in ("rouge", "all"):
        evaluate_rouge()

    if args.section in ("error", "all"):
        evaluate_error_analysis()

    print("\n" + "═" * 60)
    print("Evaluation complete.")
    print("═" * 60)


if __name__ == "__main__":
    main()
