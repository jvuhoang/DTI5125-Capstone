"""
rag_pipeline.py — PICOS-Aware Retriever + RAG Answer Generator
===============================================================
Two classes used at runtime by the chatbot:

  PICOSRetriever      — encodes a user query, searches FAISS, fetches full
                        PICOS records from SQLite, optionally filters by disease

  RAGAnswerGenerator  — builds a PICOS-structured prompt with retrieved
                        abstracts and calls the Claude API to generate a
                        grounded, cited answer

Usage:
    from rag_pipeline import PICOSRetriever, RAGAnswerGenerator

    retriever = PICOSRetriever()
    rag       = RAGAnswerGenerator(retriever)

    result = rag.answer("What interventions have been studied for Parkinson's tremor?")
    print(result["answer"])
    print(result["sources"])
"""

import sqlite3
import pickle
import numpy as np
import os
import faiss
from sentence_transformers import SentenceTransformer

DB_PATH         = "abstracts.db"
FAISS_PATH      = "abstracts.faiss"
MAP_PATH        = "faiss_id_map.pkl"
MODEL_NAME      = "all-MiniLM-L6-v2"
LOCAL_MODEL_DIR = "./models/sentence_bert"   # written by phase4_rag.py; load from here at runtime

DISEASES   = ["Alzheimer", "Parkinson", "ALS", "Huntington", "Dementia", "Stroke"]


# ── PICOSRetriever ────────────────────────────────────────────────────────────

class PICOSRetriever:
    """
    Semantic retriever that searches a FAISS index built from PICOS-enriched
    abstract embeddings. Optionally filters results by disease label.
    """

    def __init__(self,
                 db_path:    str = DB_PATH,
                 faiss_path: str = FAISS_PATH,
                 map_path:   str = MAP_PATH):
        # Load from local disk (written by phase4_rag.py) — no HuggingFace download at startup
        model_source = LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else MODEL_NAME
        if model_source == LOCAL_MODEL_DIR:
            print(f"[PICOSRetriever] Loading Sentence-BERT from local cache: {LOCAL_MODEL_DIR}")
        else:
            print(f"[PICOSRetriever] Downloading Sentence-BERT: {MODEL_NAME} (run phase4_rag.py to cache locally)")
        self.model = SentenceTransformer(model_source)

        if not os.path.exists(faiss_path):
            raise FileNotFoundError(
                f"FAISS index not found: {faiss_path}\n"
                "Run phase4_rag.py first to build the index."
            )
        self.index = faiss.read_index(faiss_path)

        with open(map_path, "rb") as f:
            self.id_map = pickle.load(f)

        # Keep a persistent connection — closed only when the object is deleted
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

    def retrieve(self,
                 query:           str,
                 k:               int        = 5,
                 filter_disease:  str        = None,
                 filter_diseases: list       = None) -> list[dict]:
        """
        Retrieve top-k PICOS-structured abstracts for a query.

        Parameters
        ----------
        query            : user question or symptom description
        k                : number of results to return per disease bucket
        filter_disease   : (legacy) single disease label to restrict results
        filter_diseases  : list of disease labels; if >1, retrieves k results
                           per disease and merges so each disease is represented

        Returns
        -------
        List of dicts with keys: pmid, title, abstract, disease, year, P, I, C, O, S
        """
        # Normalise to a list for uniform handling
        diseases = filter_diseases or ([filter_disease] if filter_disease else [])

        if len(diseases) > 1:
            # Multi-disease: fetch k abstracts per disease and merge
            seen_pmids = set()
            merged     = []
            for disease in diseases:
                bucket = self._retrieve_single(query, k=k, filter_disease=disease)
                for rec in bucket:
                    if rec["pmid"] not in seen_pmids:
                        seen_pmids.add(rec["pmid"])
                        merged.append(rec)
            return merged
        else:
            single = diseases[0] if diseases else None
            return self._retrieve_single(query, k=k, filter_disease=single)

    def _retrieve_single(self,
                         query:          str,
                         k:              int  = 5,
                         filter_disease: str  = None) -> list[dict]:
        """Core FAISS + SQLite retrieval for a single disease filter (or None)."""
        query_vec  = self.model.encode([query]).astype("float32")
        # Over-fetch to leave room after filtering
        _, indices = self.index.search(query_vec, max(k * 6, 40))

        results = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.id_map):
                continue
            db_id = self.id_map[idx]
            row   = self.conn.execute("""
                SELECT pmid, title, abstract, disease_label, year,
                       picos_population, picos_intervention,
                       picos_comparison, picos_outcome, picos_study_design
                FROM abstracts
                WHERE id = ?
            """, (db_id,)).fetchone()

            if not row:
                continue
            if filter_disease and row[3] != filter_disease:
                continue

            results.append({
                "pmid":     row[0],
                "title":    row[1],
                "abstract": row[2],
                "disease":  row[3],
                "year":     row[4],
                "P":        row[5] or "not reported",
                "I":        row[6] or "not reported",
                "C":        row[7] or "not reported",
                "O":        row[8] or "not reported",
                "S":        row[9] or "not reported",
            })
            if len(results) >= k:
                break

        return results

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass


# ── RAGAnswerGenerator ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a clinical literature assistant specialising in neurodegenerative
and neurological diseases (Alzheimer's, Parkinson's, ALS, Huntington's,
Dementia, and Stroke).

Your answers must:
- Be grounded ONLY in the provided PICOS-structured abstracts
- Cite abstract numbers in square brackets, e.g. [1], [2]
- State clearly when the provided abstracts do not contain enough information
- Never speculate beyond what the literature states
- End with a "Papers that informed this answer:" section listing each cited abstract
"""


def _build_context(abstracts: list[dict]) -> str:
    """Format retrieved abstracts into a PICOS-structured context block."""
    parts = []
    for i, a in enumerate(abstracts):
        parts.append(
            f"[{i+1}] {a['title']} ({a['disease']}, {a['year']})\n"
            f"  P (Population):   {a['P']}\n"
            f"  I (Intervention): {a['I']}\n"
            f"  C (Comparison):   {a['C']}\n"
            f"  O (Outcome):      {a['O']}\n"
            f"  S (Study design): {a['S']}\n"
            f"  Abstract excerpt: {a['abstract'][:350]}..."
        )
    return "\n\n".join(parts)


# ── Clinical knowledge base — symptoms per disease ────────────────────────────
# Used to give a direct, natural-language answer for symptom questions before
# backing it up with the retrieved literature.

_DISEASE_SYMPTOMS = {
    "Alzheimer's Disease": {
        "Early signs": [
            "memory loss, especially for recent events",
            "difficulty finding the right words",
            "losing or misplacing objects",
            "trouble with planning, problem-solving, or following steps",
            "confusion about time, dates, or familiar places",
            "mood changes — depression, anxiety, or withdrawal",
        ],
        "As it progresses": [
            "worsening memory, including forgetting close family members",
            "difficulty with daily tasks like cooking or managing finances",
            "personality and behaviour changes",
            "hallucinations or delusions",
            "wandering",
        ],
        "Late stage": [
            "loss of verbal communication",
            "difficulty swallowing",
            "loss of mobility",
            "full-time care required",
        ],
    },
    "Parkinson's Disease": {
        "Motor symptoms": [
            "resting tremor (shaking when the limb is relaxed)",
            "bradykinesia — slowness of movement",
            "muscle rigidity or stiffness",
            "postural instability and balance problems",
            "shuffling gait and reduced arm swing",
            "micrographia — small, cramped handwriting",
            "masked face — reduced facial expression",
        ],
        "Non-motor symptoms": [
            "sleep disturbances, including REM sleep behaviour disorder",
            "loss of sense of smell (anosmia)",
            "constipation and digestive issues",
            "depression and anxiety",
            "cognitive changes and, in later stages, dementia",
            "fatigue",
            "low blood pressure on standing (orthostatic hypotension)",
        ],
    },
    "ALS and Huntington's Disease": {
        "ALS symptoms": [
            "progressive muscle weakness, often starting in one hand or foot",
            "muscle twitching (fasciculations) and cramping",
            "slurred or slow speech",
            "difficulty swallowing (dysphagia)",
            "shortness of breath as breathing muscles weaken",
            "muscle wasting (atrophy)",
        ],
        "Huntington's symptoms": [
            "involuntary, irregular jerking movements (chorea)",
            "impaired balance and coordination",
            "cognitive decline — difficulty concentrating and planning",
            "psychiatric symptoms: depression, irritability, impulsivity",
            "difficulty swallowing and weight loss",
            "slurred speech",
        ],
    },
    "Dementia and Mild Cognitive Impairment": {
        "Mild Cognitive Impairment (MCI)": [
            "noticeable memory lapses beyond normal ageing",
            "occasional difficulty finding words",
            "forgetting appointments or recent conversations",
            "mostly able to manage daily life independently",
        ],
        "Dementia": [
            "significant memory loss affecting daily function",
            "confusion and disorientation to time and place",
            "difficulty with language and communication",
            "impaired judgement and decision-making",
            "personality and behavioural changes",
            "loss of independence in daily activities",
        ],
    },
    "Stroke": {
        "Acute warning signs (FAST)": [
            "Face drooping on one side",
            "Arm weakness — inability to raise both arms",
            "Speech difficulty — slurred or strange speech",
            "Time to call emergency services",
        ],
        "Other sudden symptoms": [
            "sudden numbness in the face, arm, or leg (especially one side)",
            "sudden confusion or trouble understanding",
            "sudden vision problems in one or both eyes",
            "sudden severe headache with no known cause",
            "sudden dizziness or loss of balance",
        ],
        "Post-stroke effects": [
            "paralysis or weakness on one side",
            "aphasia — difficulty speaking or understanding language",
            "memory and cognitive problems",
            "emotional changes — depression, anxiety",
            "fatigue",
        ],
    },
}


# ── Clinical diagnosis notes — how each disease is identified ────────────────

_DIAGNOSIS_NOTES = {
    "Alzheimer's Disease": (
        "**How it's clinically diagnosed:** There is no single test for Alzheimer's. "
        "A neurologist or geriatrician evaluates cognitive function using standardised tools "
        "(MMSE, MoCA), examines brain imaging (MRI or PET) for atrophy or amyloid plaques, "
        "and may order blood biomarkers (p-tau181, amyloid-β42/40 ratio). "
        "Other treatable causes of memory loss are ruled out first."
    ),
    "Parkinson's Disease": (
        "**How it's clinically diagnosed:** There's no definitive blood test. "
        "A neurologist makes a clinical diagnosis by looking for at least two of the four "
        "cardinal signs: resting tremor, bradykinesia, rigidity, and postural instability. "
        "DaTscan imaging (dopamine transporter scan) can support the diagnosis. "
        "A clear positive response to levodopa therapy also helps confirm Parkinson's."
    ),
    "ALS and Huntington's Disease": (
        "**How ALS is diagnosed:** Nerve conduction studies and EMG (electromyography) are "
        "combined with clinical examination for both upper and lower motor neuron signs. "
        "MRI is used to rule out other conditions. "
        "**How Huntington's is diagnosed:** Confirmed by a genetic blood test detecting "
        "the expanded CAG repeat in the HTT gene. Genetic counselling is strongly recommended."
    ),
    "Dementia and Mild Cognitive Impairment": (
        "**How it's clinically diagnosed:** Diagnosis involves cognitive testing (MMSE, MoCA), "
        "blood tests to rule out reversible causes (thyroid, B12, infections), "
        "and brain imaging (MRI or CT). For Alzheimer's-type dementia, PET scanning or "
        "CSF biomarkers may confirm amyloid pathology. "
        "A neuropsychologist or geriatrician typically leads the evaluation."
    ),
    "Stroke": (
        "**How it's diagnosed:** Stroke is a medical emergency — brain imaging (CT or MRI) "
        "at the hospital is required immediately to confirm the diagnosis and determine "
        "whether it is ischaemic (blocked artery) or haemorrhagic (bleeding). "
        "Time is critical: the FAST signs (Face drooping, Arm weakness, Speech difficulty — "
        "Time to call emergency services) help identify stroke quickly."
    ),
}


def _get_diagnosis_note(diseases: list) -> str:
    """Return a clinical diagnosis blurb for the named disease(s)."""
    parts = []
    for disease in diseases:
        note = _DIAGNOSIS_NOTES.get(disease, "")
        if note:
            parts.append(note)
    return "\n\n".join(parts)


def _get_symptom_answer(question: str, diseases: list) -> str:
    """
    Return a direct, natural-language symptom overview for the named disease(s).
    Returns an empty string if no matching disease is found.
    """
    q = question.lower()

    # Find which disease(s) the question is asking about
    target_diseases = []
    for label, symptoms in _DISEASE_SYMPTOMS.items():
        label_lower = label.lower()
        # Check against both the canonical label and common abbreviations in the question
        if any(word in q for word in label_lower.split()) or label in diseases:
            target_diseases.append((label, symptoms))

    if not target_diseases:
        return ""

    parts = []
    for label, symptom_groups in target_diseases:
        parts.append(f"### Symptoms of {label}\n")
        for group_name, symptom_list in symptom_groups.items():
            parts.append(f"**{group_name}:**")
            for s in symptom_list:
                parts.append(f"- {s}")
            parts.append("")

    return "\n".join(parts)


def _clean(text: str, max_len: int = 160) -> str:
    """Trim a PICOS field to a readable sentence length."""
    if not text or text.strip().lower() in ("not reported", ""):
        return ""
    t = text.strip().rstrip(".")
    return (t[:max_len] + "…") if len(t) > max_len else t


def _is_raw_sentence(text: str) -> bool:
    """
    Returns True when a PICOS field looks like a raw abstract sentence
    rather than a clean named concept (e.g. a drug name or short phrase).
    Heuristic: longer than 90 chars, or starts with a number/article.
    """
    if not text:
        return False
    t = text.strip()
    if len(t) > 90:
        return True
    if t[0].isdigit():
        return True
    if t.lower().startswith(("the ", "a ", "an ", "we ", "this ", "in ", "between ")):
        return True
    return False


def _clean_title(title: str, max_len: int = 80) -> str:
    """Return a shortened, lowercase version of a paper title for use in prose."""
    t = title.strip().rstrip(".")
    if len(t) > max_len:
        t = t[:max_len].rsplit(" ", 1)[0] + "…"
    return t


def _build_study_sentence(a: dict, idx: int, intent: str) -> str:
    """
    Write a single natural-language prose sentence describing what paper [idx]
    found, tailored to the question intent.  Uses title + clean PICOS fields;
    never dumps raw abstract text as if it were a synthesised claim.
    """
    title   = _clean_title(a.get("title", ""), max_len=80)
    excerpt = _clean(a.get("abstract", ""), max_len=180)
    I_raw   = _clean(a["I"])
    O_raw   = _clean(a["O"])
    P_raw   = _clean(a["P"])
    S_raw   = _clean(a["S"], max_len=60)

    # Only use PICOS fields when they are clean named concepts, not raw sentences
    I = I_raw if I_raw and not _is_raw_sentence(I_raw) else ""
    O = O_raw if O_raw and not _is_raw_sentence(O_raw) else ""
    P = P_raw if P_raw and not _is_raw_sentence(P_raw) else ""
    S = S_raw if S_raw and not _is_raw_sentence(S_raw) else ""

    # Fallback text: short excerpt displayed as a quote, or just the title
    fallback = f'*"{excerpt}"*' if excerpt else f"*{title}*"

    # For outcome: even when the full O_raw was too long to use as a clean concept,
    # a truncated version (≤100 chars) is still better than nothing in the treatment path.
    O_short = ""
    if O_raw and not O:
        truncated = _clean(O_raw, max_len=110)
        if truncated and not truncated.lower().startswith(
            ("the ", "a ", "an ", "we ", "this ", "in ", "between ")
        ):
            O_short = truncated

    if intent == "treatment":
        if I and O:
            s = f"**{I}** was studied"
            if P:
                s += f" in {P}"
            s += f", showing that {O.lower()} [{idx}]"
            if S:
                s += f" *({S})*"
            return s + "."
        elif I and O_short:
            s = f"**{I}** was studied"
            if P:
                s += f" in {P}"
            s += f", with findings that {O_short.lower()} [{idx}]"
            if S:
                s += f" *({S})*"
            return s + "."
        elif I:
            s = f"**{I}** was studied [{idx}]"
            if P:
                s += f" in {P}"
            return s + "."
        elif O:
            return f"Research [{idx}] found that {O.lower()}."
        else:
            return f"A study [{idx}] on *{title}* reported: {fallback}."

    elif intent in ("comparison", "diagnosis"):
        if O:
            s = f"Research [{idx}] found that {O.lower()}"
            if P:
                s += f", studied in {P}"
            return s + "."
        elif O_short:
            return f"Research [{idx}] found that {O_short.lower()}."
        else:
            return f"A study [{idx}] on *{title}* noted: {fallback}."

    elif intent == "risk":
        if O:
            s = f"Evidence [{idx}] suggests that {O.lower()}"
            if P:
                s += f" (studied in {P})"
            return s + "."
        elif O_short:
            return f"Evidence [{idx}] suggests that {O_short.lower()}."
        else:
            return f"A study [{idx}] on *{title}* found: {fallback}."

    elif intent == "progression":
        if O:
            s = f"A study [{idx}] tracking {P if P else 'patients'}"
            s += f" found that {O.lower()}."
            if S:
                s += f" *({S})*"
            return s
        elif O_short:
            return f"Research [{idx}] found that {O_short.lower()}."
        else:
            return f"Research [{idx}] on *{title}* reported: {fallback}."

    else:  # general
        if O:
            return f"Research [{idx}] found that {O.lower()}."
        elif I:
            return f"**{I}** was studied [{idx}]" + (f" in {P}." if P else ".")
        else:
            return f"A study [{idx}] on *{title}* reported: {fallback}."


def _build_lit_support_sentence(a: dict, idx: int) -> str:
    """
    Write a supporting sentence for the literature section of symptom answers.
    Uses title + clean outcome; falls back to an attributed excerpt quote.
    """
    title   = _clean_title(a.get("title", ""), max_len=90)
    year    = a.get("year", "recent")
    O       = _clean(a["O"])
    P       = _clean(a["P"])
    excerpt = _clean(a.get("abstract", ""), max_len=170)

    # Use outcome if it's a clean finding, not a raw sentence
    if O and not _is_raw_sentence(O):
        s = f"Research on *{title}* [{idx}] found that {O.lower()}"
        if P and not _is_raw_sentence(P):
            s += f", in {P}"
        return s + "."
    # Otherwise use a short excerpt as a quote
    if excerpt:
        return f"A {year} study on *{title}* [{idx}] reported: *\"{excerpt}\"*"
    return f"See [{idx}] *{title}*."


# ── Clinical context for common comparison questions ─────────────────────────
_COMPARISON_CONTEXT = {
    ("dementia", "parkinson"):
        "Dementia is a broad syndrome — an umbrella term for symptoms of cognitive "
        "decline — while Parkinson's disease is a specific neurodegenerative disorder "
        "primarily affecting movement. That said, they can overlap: up to 80% of people "
        "with Parkinson's develop dementia over time (called Parkinson's Disease Dementia). "
        "The key distinction is *what comes first* — in Parkinson's, motor symptoms "
        "(tremor, rigidity, slow movement) appear before cognitive ones, whereas in "
        "Alzheimer's and most other dementias, cognitive decline is the first sign.",

    ("alzheimer", "dementia"):
        "Alzheimer's disease is actually *a type* of dementia — dementia is the umbrella "
        "term for cognitive decline syndromes, and Alzheimer's is the most common cause "
        "(accounting for 60–80% of cases). Other types include vascular dementia, Lewy "
        "body dementia, and frontotemporal dementia. Distinguishing which type you have "
        "matters because treatment approaches and progression differ.",

    ("alzheimer", "parkinson"):
        "Both are neurodegenerative diseases, but they affect different systems first. "
        "Alzheimer's primarily targets memory and cognition early on, caused by amyloid "
        "plaques and tau tangles. Parkinson's primarily targets movement via loss of "
        "dopamine-producing neurons, causing tremor, rigidity, and slowness. Both can "
        "eventually affect cognition and behaviour, but the starting profile is distinct.",

    ("als", "parkinson"):
        "ALS (amyotrophic lateral sclerosis) and Parkinson's are both neurodegenerative "
        "but affect very different systems. ALS attacks motor neurons controlling voluntary "
        "muscles, leading to progressive paralysis. Parkinson's affects the dopamine system, "
        "primarily causing movement symptoms like tremor and rigidity. ALS progresses faster "
        "and currently has no disease-modifying treatment.",
}


def _get_comparison_context(question: str, diseases: list[str]) -> str:
    """Return a pre-written clinical context blurb if the question compares two known diseases."""
    q = question.lower()
    disease_lower = [d.lower() for d in diseases]
    for (d1, d2), blurb in _COMPARISON_CONTEXT.items():
        if (d1 in q or any(d1 in dl for dl in disease_lower)) and \
           (d2 in q or any(d2 in dl for dl in disease_lower)):
            return blurb
    return ""


def _generate_picos_answer(question: str, abstracts: list[dict]) -> str:
    """
    Generate a conversational, synthesized answer from retrieved PICOS abstracts.
    Detects question intent and writes flowing prose with inline citations,
    followed by a compact source list.  No API key required.
    """
    q = question.lower()

    # ── Intent detection ──────────────────────────────────────────────────────
    is_comparison = any(w in q for w in [
        "difference", "differences", "compare", "comparing", "contrast",
        "distinguish", "different from", "similar to", "versus", " vs ",
        "what separates", "what's the difference",
    ])
    # "how do I know if I have X" → treat as symptom question (add diagnosis note)
    is_know_if    = any(p in q for p in [
        "how do i know", "how would i know", "how can i tell",
        "tell if i have", "know if i have", "how do you know",
    ])
    is_symptom_q = is_know_if or any(w in q for w in [
        "symptom", "sign", "feel", "experience", "what happens", "indicate",
        "manifest", "present", "look like", "symptoms of",
    ])
    is_diagnosis = is_comparison or (not is_symptom_q and any(w in q for w in [
        "diagnos", "or dementia", "or alzheimer", "or parkinson",
        "confirm", "test for", "detected",
    ]))
    is_treatment = any(w in q for w in [
        "treatment", "treat", "therapy", "drug", "medication", "manage",
        "intervention", "help with", "prescribed", "given for",
    ])
    is_risk = any(w in q for w in [
        "risk", "cause", "factor", "likely", "prevent", "avoid",
        "predispos", "linked to", "associated with",
    ])
    is_progression = any(w in q for w in [
        "progression", "get worse", "worsen", "prognosis", "long term",
        "survival", "life expectancy", "stage", "advance",
    ])
    # Default to treatment if nothing else matched
    if not any([is_comparison, is_symptom_q, is_diagnosis,
                is_treatment, is_risk, is_progression]):
        is_treatment = True

    n = len(abstracts)
    diseases = sorted({a["disease"] for a in abstracts})
    disease_str = " and ".join(diseases[:2]) if diseases else "neurological conditions"

    # ── Compact, numbered source list (shared by all paths) ───────────────────
    def _source_block(items):
        lines = ["\n\n---\n**Sources:**"]
        for i, a in items:
            t = a["title"]
            td = (t[:78] + "…") if len(t) > 78 else t
            lines.append(f"[{i}] {td} ({a['year']}) — PMID {a['pmid']}")
        return "\n".join(lines)

    indexed = list(enumerate(abstracts, 1))

    # ── SYMPTOM path: clinical knowledge first, literature support after ──────
    if is_symptom_q and not is_comparison:
        symptom_answer = _get_symptom_answer(question, diseases)
        if symptom_answer:
            # Optional "how it's diagnosed" section for "how do I know" questions
            diag_note = ""
            if is_know_if:
                diag_note_text = _get_diagnosis_note(diseases)
                if diag_note_text:
                    diag_note = f"\n\n---\n{diag_note_text}"

            # Literature support — one sentence per paper, using title + clean PICOS
            lit_items = []
            for i, a in indexed:
                sentence = _build_lit_support_sentence(a, i)
                if sentence:
                    lit_items.append(sentence)

            lit_section = ""
            if lit_items:
                lit_section = (
                    "\n\n---\n**What the research literature adds:**\n\n"
                    + "\n\n".join(lit_items[:4])
                )

            closing = (
                "\n\n> ⚕️ Symptoms vary between individuals and by stage. "
                "Only a qualified clinician can provide a formal assessment."
            )

            intro = (
                f"Here's an overview of the symptoms associated with {disease_str}:\n\n"
                if not is_know_if else
                f"Here's what to look out for with {disease_str}:\n\n"
            )

            return (
                intro
                + symptom_answer
                + diag_note
                + lit_section
                + _source_block(indexed)
                + closing
            )

    # ── COMPARISON path: clinical blurb first, then per-paper synthesis ───────
    if is_comparison:
        comparison_blurb = _get_comparison_context(question, diseases)
        opening = (
            f"Great question — these are often confused. {comparison_blurb}\n\n"
            f"Here's what recent studies in the literature specifically found:\n"
            if comparison_blurb else
            f"These conditions share some overlap but have important distinctions. "
            f"Here's what {n} studies in the literature found:\n"
        )
        intent_key = "comparison"

    # ── TREATMENT path ────────────────────────────────────────────────────────
    elif is_treatment:
        opening = (
            f"Several approaches have been studied for {disease_str}. "
            f"Here's a synthesis of what the literature shows:\n"
        )
        intent_key = "treatment"

    # ── RISK path ─────────────────────────────────────────────────────────────
    elif is_risk:
        opening = (
            f"Several risk factors have been identified for {disease_str}. "
            f"Here's what the evidence says across {n} studies:\n"
        )
        intent_key = "risk"

    # ── PROGRESSION path ──────────────────────────────────────────────────────
    elif is_progression:
        opening = (
            f"Understanding how {disease_str} progresses is an active area of research. "
            f"Here's what {n} studies suggest:\n"
        )
        intent_key = "progression"

    # ── DIAGNOSIS (explicit) path ─────────────────────────────────────────────
    elif is_diagnosis:
        opening = (
            f"Diagnosis of {disease_str} involves several steps. "
            f"Here's what recent studies in the literature show:\n"
        )
        intent_key = "diagnosis"

    # ── GENERAL / fallback path ───────────────────────────────────────────────
    else:
        opening = (
            f"Based on {n} studies from the PubMed literature on {disease_str}, "
            f"here's what the research shows:\n"
        )
        intent_key = "general"

    # ── Body — one synthesised prose sentence per paper ───────────────────────
    body_lines = []
    for i, a in indexed:
        sentence = _build_study_sentence(a, i, intent_key)
        if sentence:
            body_lines.append(sentence)

    body = "\n\n".join(body_lines) if body_lines else (
        "The retrieved studies did not contain enough specific information to "
        "answer this question directly. Try rephrasing or broadening your query."
    )

    # ── Closing note ─────────────────────────────────────────────────────────
    if is_comparison or is_diagnosis:
        closing = (
            "\n\n> ⚕️ **Important:** A definitive diagnosis always requires a qualified "
            "clinician. Workup typically includes cognitive screening (MMSE, MoCA), "
            "neurological examination, blood tests, and brain imaging (MRI or PET)."
        )
    else:
        closing = (
            "\n\n*For clinical decisions, please consult a qualified healthcare professional.*"
        )

    return opening + "\n\n" + body + closing + _source_block(indexed)


class RAGAnswerGenerator:
    """
    Retrieves PICOS-structured abstracts and generates a grounded answer.

    If ANTHROPIC_API_KEY is set in the environment, uses the Claude API for
    a fluent natural-language response.  Otherwise falls back to a structured
    PICOS summary built directly from the retrieved abstracts — no API key required.
    """

    def __init__(self, retriever: PICOSRetriever):
        self.retriever = retriever
        # Try to set up the Claude client — will be None if no API key
        self.client = None
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except Exception as e:
                print(f"[WARN] Could not initialise Anthropic client: {e}. "
                      "Falling back to PICOS template answers.")

    def answer(self,
               question:        str,
               k:               int   = 5,
               filter_disease:  str   = None,
               filter_diseases: list  = None) -> dict:
        """
        Generate a grounded answer for a clinical literature question.

        Parameters
        ----------
        filter_diseases : list of disease labels — when >1, retrieves k abstracts
                          per disease so every mentioned condition is covered.
        filter_disease  : (legacy) single disease label; ignored if filter_diseases set.

        Returns
        -------
        dict with keys:
          answer        : str — full response
          sources       : list of "PMID XXXXXXXX" strings
          picos_summary : list of dicts with pmid, title, year, P, I, O, S
        """
        abstracts = self.retriever.retrieve(
            question,
            k=k,
            filter_disease=filter_disease,
            filter_diseases=filter_diseases,
        )

        if not abstracts:
            return {
                "answer":        "No relevant literature found in the database "
                                 "for this query. Please try rephrasing or broadening "
                                 "your question.",
                "sources":       [],
                "picos_summary": [],
            }

        # Use Claude API if available, otherwise use the free PICOS template generator
        if self.client is not None:
            try:
                context    = _build_context(abstracts)
                user_prompt = (
                    f"RETRIEVED ABSTRACTS (PICOS-structured):\n{context}\n\n"
                    f"QUESTION: {question}\n\nANSWER:"
                )
                message = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=700,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                answer_text = message.content[0].text
            except Exception as e:
                print(f"[WARN] Claude API call failed: {e}. Using PICOS template.")
                answer_text = _generate_picos_answer(question, abstracts)
        else:
            answer_text = _generate_picos_answer(question, abstracts)

        return {
            "answer":  answer_text,
            "sources": [f"PMID {a['pmid']}" for a in abstracts],
            "picos_summary": [
                {
                    "pmid":  a["pmid"],
                    "title": a["title"],
                    "year":  a["year"],
                    "P":     a["P"],
                    "I":     a["I"],
                    "O":     a["O"],
                    "S":     a["S"],
                }
                for a in abstracts
            ],
        }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = PICOSRetriever()
    rag       = RAGAnswerGenerator(retriever)

    test_questions = [
        "What interventions have been studied for Parkinson's tremor?",
        "What are the outcomes of levodopa therapy in Parkinson's disease?",
        "Which clinical trials studied cognitive decline in Alzheimer's patients?",
        "What treatments exist for ALS progression?",
        "What are the risk factors for stroke in older adults?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = rag.answer(q, k=3)
        print(f"A: {result['answer'][:400]}...")
        print(f"Sources: {result['sources']}")
