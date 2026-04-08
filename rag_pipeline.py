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
- Ask users if they have any other questions and give 3 similar questions as suggestions
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


# ── Clinical treatments knowledge base ───────────────────────────────────────

_DISEASE_TREATMENTS = {
    "Alzheimer's Disease": {
        "Disease-modifying therapies (FDA-approved)": [
            "lecanemab (Leqembi) — anti-amyloid antibody shown to slow cognitive decline "
            "in early-stage Alzheimer's",
            "donanemab (Kisunla) — another anti-amyloid antibody; slows decline in "
            "early symptomatic stages",
            "aducanumab (Aduhelm) — approved but less widely used; removes amyloid plaques",
        ],
        "Symptomatic medications": [
            "cholinesterase inhibitors — donepezil (Aricept), rivastigmine, galantamine; "
            "improve memory and thinking in mild-to-moderate Alzheimer's",
            "memantine (Namenda) — for moderate-to-severe Alzheimer's; regulates glutamate activity",
        ],
        "Non-pharmacological approaches": [
            "cognitive stimulation therapy and structured activities",
            "physical exercise — shown to slow cognitive decline",
            "management of cardiovascular risk factors (blood pressure, diabetes, cholesterol)",
            "social engagement and mental stimulation",
            "caregiver support programmes",
        ],
    },
    "Parkinson's Disease": {
        "First-line medications": [
            "levodopa/carbidopa (Sinemet) — gold standard; replaces dopamine and greatly "
            "reduces motor symptoms",
            "dopamine agonists — pramipexole, ropinirole, rotigotine patch; used in early "
            "disease or alongside levodopa",
            "MAO-B inhibitors — selegiline, rasagiline, safinamide; prolong dopamine action",
        ],
        "For advanced Parkinson's": [
            "deep brain stimulation (DBS) — electrical stimulation of specific brain areas; "
            "reduces tremor, rigidity, and dyskinesia significantly",
            "levodopa intestinal gel (Duodopa) — continuous infusion for fluctuating symptoms",
            "COMT inhibitors — entacapone, opicapone; reduce 'off' periods",
        ],
        "Non-pharmacological": [
            "physiotherapy — improves balance, gait, and falls prevention",
            "speech and language therapy — for voice and swallowing difficulties",
            "occupational therapy — maintains independence in daily tasks",
            "exercise programmes (e.g., tai chi, boxing) — shown to slow motor decline",
        ],
    },
    "ALS and Huntington's Disease": {
        "ALS — disease-modifying treatments": [
            "riluzole — modestly slows progression by reducing glutamate toxicity; extends "
            "survival by approximately 2–3 months",
            "edaravone (Radicava) — reduces oxidative stress; may slow functional decline "
            "in a subset of patients",
            "tofersen — for ALS caused by SOD1 gene mutations; reduces SOD1 protein levels",
        ],
        "ALS — supportive care": [
            "non-invasive ventilation (BiPAP/CPAP) — as breathing muscles weaken",
            "percutaneous endoscopic gastrostomy (PEG) — feeding tube when swallowing fails",
            "communication aids and augmentative communication devices",
            "multidisciplinary clinic care — neurologist, physio, speech, dietitian, palliative",
        ],
        "Huntington's Disease — symptom management": [
            "tetrabenazine or deutetrabenazine — reduce involuntary movements (chorea)",
            "antidepressants (SSRIs) — for depression and irritability",
            "antipsychotics — for psychosis and agitation",
            "speech and swallowing therapy",
            "no disease-modifying treatment currently approved; trials ongoing",
        ],
    },
    "Dementia and Mild Cognitive Impairment": {
        "Medications for Alzheimer's-type dementia": [
            "cholinesterase inhibitors (donepezil, rivastigmine) — for mild-to-moderate dementia",
            "memantine — for moderate-to-severe dementia",
            "anti-amyloid antibodies (lecanemab, donanemab) — if early Alzheimer's confirmed",
        ],
        "Managing other dementia types": [
            "vascular dementia — managing blood pressure, cholesterol, and diabetes; "
            "antiplatelet therapy to prevent further strokes",
            "Lewy body dementia — rivastigmine for cognition; caution with antipsychotics "
            "(can cause severe reactions)",
            "frontotemporal dementia — no approved drugs; manage behavioural symptoms carefully",
        ],
        "Non-pharmacological": [
            "structured cognitive stimulation and reminiscence therapy",
            "physical activity programmes",
            "carer education and support",
            "safe home environment to reduce falls and wandering risks",
        ],
    },
    "Stroke": {
        "Acute treatment (emergency)": [
            "intravenous thrombolysis — tPA (alteplase) given within 4.5 hours of ischaemic "
            "stroke onset to dissolve the blood clot",
            "mechanical thrombectomy — catheter-based clot removal; effective up to 24 hours "
            "in selected patients",
            "blood pressure control and surgery for haemorrhagic stroke",
        ],
        "Secondary prevention": [
            "antiplatelet therapy — aspirin, clopidogrel; prevents recurrent ischaemic stroke",
            "anticoagulation — warfarin or DOACs (apixaban, rivaroxaban) for atrial fibrillation",
            "statins — lower cholesterol and reduce stroke recurrence risk",
            "blood pressure management — ACE inhibitors, ARBs",
        ],
        "Rehabilitation": [
            "physiotherapy — regain mobility and strength",
            "speech and language therapy — for aphasia and swallowing difficulties",
            "occupational therapy — relearn daily living skills",
            "neuropsychological support — for cognitive and emotional changes",
            "early intensive rehabilitation improves long-term outcomes",
        ],
    },
}


# ── Prevention and risk-reduction knowledge base ─────────────────────────────

_DISEASE_PREVENTION = {
    "Alzheimer's Disease": {
        "Lifestyle interventions with strongest evidence": [
            "regular aerobic exercise — reduces amyloid burden, improves neuroplasticity; "
            "150+ minutes per week recommended",
            "cognitive engagement — lifelong learning, mentally stimulating activities, "
            "bilingualism, and new skill acquisition build cognitive reserve",
            "Mediterranean or MIND diet — high in vegetables, fish, olive oil, nuts; "
            "associated with 35–53% lower risk",
            "social connection — active social life and strong relationships are consistently "
            "associated with lower dementia risk",
            "quality sleep — sleep clears amyloid from the brain; 7–8 hours per night; "
            "treat sleep apnoea if present",
        ],
        "Cardiovascular risk management": [
            "control blood pressure — hypertension in midlife is one of the largest "
            "modifiable risk factors; target systolic <130 mmHg",
            "manage diabetes — tight glycaemic control reduces brain damage from vascular events",
            "lower high cholesterol — statins may reduce vascular dementia risk",
            "maintain healthy weight — obesity in midlife is associated with higher risk",
            "quit smoking — smoking cessation reduces risk over time",
        ],
        "Sensory and mental health": [
            "treat hearing loss early — hearing aids are associated with reduced dementia risk; "
            "hearing loss is the single largest modifiable risk factor (2020 Lancet Commission)",
            "manage depression — depression is both a risk factor and an early symptom; "
            "effective treatment may lower risk",
            "reduce social isolation — loneliness accelerates cognitive decline",
        ],
        "Currently investigational / emerging": [
            "avoiding chronic stress (linked to cortisol-related hippocampal damage)",
            "B vitamin supplementation in those with elevated homocysteine",
            "moderate alcohol consumption or abstinence (heavy drinking increases risk)",
        ],
    },
    "Parkinson's Disease": {
        "Physical activity": [
            "regular aerobic exercise — the strongest evidence-based protective factor; "
            "associated with 30–40% lower risk in several cohort studies",
            "vigorous exercise (running, cycling, dancing) appears most protective",
        ],
        "Diet and lifestyle": [
            "caffeine consumption — coffee and tea intake consistently associated with "
            "20–30% lower risk; mechanism may involve adenosine receptor antagonism",
            "avoid pesticide exposure — herbicides (paraquat, rotenone) and insecticides "
            "are the most established environmental risk factor",
            "avoid heavy metals — occupational manganese and lead exposure increases risk",
            "Mediterranean diet — may offer modest protective benefit",
        ],
        "Medical": [
            "ibuprofen use — some epidemiological studies suggest regular NSAID use is "
            "associated with lower risk (not a recommended strategy without medical guidance)",
            "oestrogen exposure — some studies suggest hormonal factors modulate risk; "
            "mechanism unclear",
        ],
        "Note": [
            "no single intervention is proven to prevent Parkinson's; "
            "exercise and avoiding pesticides have the strongest evidence",
        ],
    },
    "ALS and Huntington's Disease": {
        "ALS — what may reduce risk": [
            "avoid smoking — the most established modifiable environmental risk factor",
            "minimise pesticide and heavy metal exposure where possible",
            "maintain physical fitness without extreme exertion — very intense "
            "long-term athletic activity has been associated with slightly higher risk",
        ],
        "ALS — limitations": [
            "for sporadic ALS (~90% of cases), there are no proven prevention strategies",
            "for familial ALS, genetic counselling and testing (C9orf72, SOD1) can inform "
            "family planning decisions",
        ],
        "Huntington's Disease": [
            "Huntington's is caused entirely by a genetic mutation — it cannot be prevented "
            "through lifestyle",
            "predictive genetic testing is available for at-risk individuals (those with an "
            "affected parent)",
            "preimplantation genetic diagnosis (PGD) allows prospective parents to have "
            "children without the mutation",
            "exercise may slow symptom progression once diagnosed but does not prevent onset",
        ],
    },
    "Dementia and Mild Cognitive Impairment": {
        "Evidence-based risk reduction (Lancet Commission 2024 — 14 modifiable factors)": [
            "control hypertension — the most impactful intervention; start in midlife",
            "treat hearing loss — hearing aids associated with slower cognitive decline",
            "increase physical activity — 150+ min/week moderate aerobic exercise",
            "manage depression",
            "maintain social engagement",
            "quit smoking",
            "limit alcohol — heavy drinking is directly neurotoxic",
            "manage diabetes and obesity",
            "reduce air pollution exposure where possible",
            "address traumatic brain injury prevention (helmets, fall prevention)",
        ],
        "Cognitive reserve building": [
            "higher education is protective — pursue lifelong learning",
            "bilingualism and learning new languages",
            "mentally stimulating work and hobbies",
        ],
        "Sleep and stress": [
            "treat sleep apnoea — untreated OSA accelerates cognitive decline",
            "target 7–8 hours of quality sleep",
            "chronic stress management — cortisol damages the hippocampus over time",
        ],
    },
    "Stroke": {
        "Blood pressure and cardiovascular control": [
            "control hypertension — the single most important intervention; "
            "every 10 mmHg reduction in systolic BP reduces stroke risk by ~30%",
            "treat atrial fibrillation — anticoagulation (warfarin, NOACs) reduces "
            "stroke risk by ~65% in AF patients",
            "manage diabetes — good glycaemic control reduces microvascular damage",
            "lower LDL cholesterol — statins reduce stroke risk by ~20–30%",
        ],
        "Lifestyle changes": [
            "quit smoking — smoking doubles stroke risk; risk normalises within 2–5 years "
            "of cessation",
            "regular physical activity — 30 min moderate exercise most days",
            "healthy diet — Mediterranean or DASH diet; reduce salt intake",
            "limit alcohol — no more than 1–2 units/day",
            "maintain healthy weight — obesity increases risk via hypertension and diabetes",
        ],
        "Secondary prevention (after a stroke or TIA)": [
            "antiplatelet therapy — aspirin or clopidogrel reduces recurrent ischaemic stroke",
            "carotid endarterectomy — for high-grade carotid stenosis (>70%)",
            "blood pressure and statin therapy continued indefinitely",
        ],
    },
}


# ── Clinical risk factors knowledge base ─────────────────────────────────────

_DISEASE_RISK_FACTORS = {
    "Alzheimer's Disease": {
        "Non-modifiable risk factors": [
            "age — the strongest risk factor; risk doubles every 5 years after 65",
            "genetics — APOE ε4 allele increases risk 3–4× (one copy) or 8–12× (two copies)",
            "family history — first-degree relative with Alzheimer's doubles your risk",
            "Down syndrome — almost all individuals develop Alzheimer's pathology by their 40s",
        ],
        "Modifiable risk factors": [
            "cardiovascular risk — hypertension, diabetes, obesity, high cholesterol",
            "physical inactivity",
            "smoking",
            "depression — both a risk factor and early symptom",
            "hearing loss — untreated hearing loss is one of the largest modifiable risks",
            "low education and limited cognitive reserve",
            "social isolation",
            "traumatic brain injury",
        ],
    },
    "Parkinson's Disease": {
        "Non-modifiable risk factors": [
            "age — risk increases significantly after 60",
            "male sex — men are approximately 1.5× more likely to develop Parkinson's",
            "genetics — mutations in LRRK2, PINK1, SNCA, PARK7 genes; familial cases account "
            "for ~10–15% of diagnoses",
        ],
        "Environmental risk factors": [
            "pesticide and herbicide exposure (e.g., rotenone, paraquat) — strongly linked",
            "heavy metal exposure (manganese, lead)",
            "well water consumption in rural settings",
            "traumatic brain injury",
        ],
        "Protective factors (lower risk)": [
            "smoking — paradoxically associated with lower risk (not recommended as a strategy)",
            "caffeine / coffee consumption",
            "regular physical exercise",
            "ibuprofen / NSAID use (some studies show association)",
        ],
    },
    "ALS and Huntington's Disease": {
        "ALS risk factors": [
            "genetics — C9orf72, SOD1, FUS, TARDBP mutations account for ~10% of cases "
            "(familial ALS); the rest are sporadic",
            "military service — veterans have 1.5–2× higher risk than civilians",
            "heavy physical labour and intense athletic activity",
            "smoking — the most established environmental risk factor",
            "age — peak onset between 55–75",
            "male sex — slightly higher risk",
        ],
        "Huntington's Disease": [
            "purely genetic — caused by CAG repeat expansion in the HTT gene; "
            "autosomal dominant inheritance (50% chance of passing to children)",
            "longer CAG repeat = earlier onset and more severe disease",
            "if a parent has Huntington's, each child has a 50% chance of inheriting it",
            "no environmental risk factors identified",
        ],
    },
    "Dementia and Mild Cognitive Impairment": {
        "Vascular and metabolic risk factors": [
            "hypertension — the most important modifiable risk factor",
            "type 2 diabetes",
            "atrial fibrillation — increases stroke risk and vascular dementia",
            "high cholesterol",
            "obesity, especially in midlife",
        ],
        "Lifestyle risk factors": [
            "physical inactivity",
            "smoking",
            "heavy alcohol consumption",
            "social isolation and loneliness",
            "untreated depression",
            "poor sleep (links to amyloid accumulation)",
        ],
        "Non-modifiable": [
            "age",
            "APOE ε4 genotype",
            "family history",
            "prior head injuries",
        ],
    },
    "Stroke": {
        "Major modifiable risk factors": [
            "hypertension — the single most important risk factor; responsible for approximately "
            "54% of strokes worldwide",
            "atrial fibrillation — 5× increased stroke risk; requires anticoagulation",
            "diabetes",
            "smoking — doubles stroke risk",
            "high cholesterol / hyperlipidaemia",
            "obesity",
            "physical inactivity",
            "excessive alcohol",
        ],
        "Non-modifiable risk factors": [
            "age — risk doubles each decade after 55",
            "male sex",
            "family history",
            "prior stroke or TIA (transient ischaemic attack)",
            "ethnicity — higher risk in Black and South Asian populations",
        ],
        "Less common causes": [
            "carotid artery disease",
            "patent foramen ovale (PFO)",
            "sleep apnoea",
            "cocaine and amphetamine use",
            "oral contraceptive pill (particularly with smoking or migraine)",
        ],
    },
}


def _build_knowledge_section(label: str, knowledge_dict: dict, section_title: str) -> str:
    """Render a structured knowledge block (symptoms / treatments / risk factors)."""
    data = knowledge_dict.get(label, {})
    if not data:
        return ""
    parts = [f"### {section_title} of {label}\n"]
    for group_name, items in data.items():
        parts.append(f"**{group_name}:**")
        for item in items:
            parts.append(f"- {item}")
        parts.append("")
    return "\n".join(parts)


def _get_treatment_answer(diseases: list) -> str:
    """Return a structured treatment overview for the named disease(s)."""
    parts = []
    for d in diseases:
        block = _build_knowledge_section(d, _DISEASE_TREATMENTS, "Treatments")
        if block:
            parts.append(block)
    return "\n".join(parts)


def _get_risk_factor_answer(diseases: list) -> str:
    """Return a structured risk factor overview for the named disease(s)."""
    parts = []
    for d in diseases:
        block = _build_knowledge_section(d, _DISEASE_RISK_FACTORS, "Risk Factors")
        if block:
            parts.append(block)
    return "\n".join(parts)


def _get_prevention_answer(diseases: list) -> str:
    """Return a structured risk-reduction and prevention overview for the named disease(s)."""
    parts = []
    for d in diseases:
        block = _build_knowledge_section(d, _DISEASE_PREVENTION, "Risk Reduction")
        if block:
            parts.append(block)
    return "\n".join(parts)


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
    rather than a clean named concept (e.g. a drug name, short phrase).
    Heuristic: longer than 80 chars OR starts with a common article/pronoun/
    discourse marker.  Kept intentionally strict to avoid using messy LLM
    extractions as if they were clean clinical terms.
    """
    if not text:
        return False
    t = text.strip()
    if len(t) > 80:
        return True
    if t[0].isdigit():
        return True
    _BAD_STARTS = (
        "the ", "a ", "an ", "we ", "this ", "in ", "between ",
        "our ", "these ", "there ", "despite ", "although ", "however ",
        "patients with ", "individuals with ", "subjects with ",
        "all ", "both ", "while ", "among ", "for ",
        "substantial ", "significant ", "overall ", "recent ", "current ",
        "results ", "findings ", "data ", "evidence ", "study ",
    )
    if t.lower().startswith(_BAD_STARTS):
        return True
    # Sentences with a verb + auxiliary pattern are likely raw abstract text
    if " were " in t.lower() or " was " in t.lower() or " were " in t.lower():
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

    # Clean title for fallback (concise, no subtitles, no brackets)
    clean_title = _make_clean_title(a.get("title", ""))
    S_label = S.lower() if S else "study"
    fallback = f"*{clean_title}* — {S_label}"

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


def _make_clean_title(title: str, max_len: int = 85) -> str:
    """
    Return a concise, readable version of a paper title for use in citations.
    Strips leading brackets (e.g., "[Huntington's disease presenting with..."),
    takes the main clause before a long subtitle colon, and truncates.
    """
    t = title.strip()
    # Remove leading [bracketed] article markers
    if t.startswith("["):
        end = t.find("]")
        if 0 < end < 80:
            t = t[end + 1:].strip(" :")
    # Keep only the main clause if there is a descriptive subtitle after ":"
    if ":" in t:
        colon = t.index(":")
        if colon > 20:
            t = t[:colon]
    # Truncate
    if len(t) > max_len:
        t = t[:max_len].rsplit(" ", 1)[0] + "…"
    return t


def _build_lit_support_sentence(a: dict, idx: int) -> str:
    """
    Write one compact, readable citation line for the 'What the literature adds'
    section. Tries clean O first; falls back to a tidy bibliography-style line.
    """
    title = _make_clean_title(a.get("title", ""))
    year  = a.get("year", "")
    O     = _clean(a["O"])
    S     = _clean(a["S"], max_len=50)

    # Only use O if it is a genuinely short, clean finding
    if O and len(O) <= 70 and not _is_raw_sentence(O):
        return f"**[{idx}]** Research found that {O.lower()}."

    # Otherwise give a clean bibliography-style reference
    study_label = (
        S.lower()
        if S and not _is_raw_sentence(S)
        else "study"
    )
    year_str = f", {year}" if year else ""
    return f"**[{idx}]** *{title}* ({study_label}{year_str})"


# ── Clinical context for common comparison questions ─────────────────────────
# Covers all 15 pairs across the 6 disease groups in the system.
# Keys use short lowercase tokens that appear in disease names or the question.
_COMPARISON_CONTEXT = {

    # ── Alzheimer's vs Parkinson's ────────────────────────────────────────────
    ("alzheimer", "parkinson"):
        "Both are progressive neurodegenerative diseases, but they attack different "
        "parts of the brain. **Alzheimer's** primarily destroys memory and cognition "
        "first — caused by amyloid plaques and tau tangles building up in the brain. "
        "**Parkinson's** primarily targets movement by killing dopamine-producing neurons "
        "in the substantia nigra, causing tremor, rigidity, and slowness. Both can "
        "eventually affect the full brain, but the *starting* symptoms are very different: "
        "memory loss first suggests Alzheimer's, tremor and stiffness first suggests Parkinson's.",

    # ── Alzheimer's vs Dementia ───────────────────────────────────────────────
    ("alzheimer", "dementia"):
        "Alzheimer's disease is actually **a type of dementia** — dementia is the umbrella "
        "term for any syndrome of progressive cognitive decline, and Alzheimer's is the most "
        "common cause (60–80% of cases). Other types include vascular dementia, Lewy body "
        "dementia, and frontotemporal dementia. Distinguishing the type matters because "
        "treatments, progression, and genetic risks differ significantly between them.",

    # ── Alzheimer's vs ALS / Huntington's ────────────────────────────────────
    ("alzheimer", "als"):
        "**Alzheimer's** and **ALS** are both progressive and incurable, but they affect "
        "completely different systems. Alzheimer's destroys memory and cognition; ALS destroys "
        "the motor neurons that control voluntary movement, leading to progressive weakness and "
        "paralysis. Alzheimer's rarely affects movement until late stages; ALS rarely affects "
        "cognition (though ~10–15% of ALS patients also develop frontotemporal dementia). "
        "ALS also progresses faster — median survival is 2–5 years from diagnosis versus "
        "8–10 years for Alzheimer's.",

    ("alzheimer", "huntington"):
        "**Alzheimer's** is a late-onset disease (usually after 65) caused by amyloid plaques "
        "and tau tangles, primarily affecting memory first. **Huntington's** is a genetic "
        "disorder (inherited from a parent) that typically appears in midlife (30s–50s) and "
        "affects movement (chorea — uncontrolled jerky movements), mood, and cognition "
        "simultaneously. A key practical difference: Huntington's can be predicted with a "
        "genetic test before any symptoms appear; Alzheimer's cannot.",

    # ── Alzheimer's vs Stroke ─────────────────────────────────────────────────
    ("alzheimer", "stroke"):
        "**Alzheimer's** and **stroke** can both cause cognitive decline, but their cause "
        "and onset are very different. Alzheimer's is a neurodegenerative disease — symptoms "
        "emerge slowly over years. Stroke is an acute vascular event (a clot or bleed in "
        "the brain) where symptoms appear suddenly — within seconds to minutes. Repeated "
        "small strokes can cause *vascular dementia*, which can look similar to Alzheimer's "
        "on the surface but shows up differently on MRI. Brain imaging is essential to "
        "tell them apart.",

    # ── Parkinson's vs Dementia ───────────────────────────────────────────────
    ("dementia", "parkinson"):
        "**Dementia** is a broad syndrome — an umbrella term for progressive cognitive "
        "decline — while **Parkinson's** is a specific disease that primarily causes movement "
        "problems (tremor, rigidity, slow movement). The two can overlap: up to 80% of people "
        "with Parkinson's develop dementia over time (Parkinson's Disease Dementia). The key "
        "distinction is *what comes first* — Parkinson's starts with motor symptoms; most "
        "other dementias (especially Alzheimer's) start with memory and cognitive symptoms.",

    # ── Parkinson's vs ALS / Huntington's ────────────────────────────────────
    ("als", "parkinson"):
        "**ALS** and **Parkinson's** are both neurodegenerative but affect very different "
        "systems. ALS destroys the motor neurons that control voluntary muscles, causing "
        "progressive paralysis and, eventually, respiratory failure. Parkinson's affects "
        "the dopamine system, causing tremor, rigidity, and slowness — but patients can "
        "live for decades with it. ALS progresses much faster and has no disease-modifying "
        "treatment, whereas Parkinson's can be well-controlled with levodopa and other drugs.",

    ("huntington", "parkinson"):
        "Both cause movement problems, but in different ways. **Huntington's** causes chorea "
        "— involuntary, jerky, dance-like movements — and is caused by a specific genetic "
        "mutation (CAG repeat expansion in the HTT gene). **Parkinson's** causes hypokinesia "
        "— slow, reduced movement, tremor, and rigidity. Huntington's is inherited and has "
        "a known genetic cause; most Parkinson's cases are sporadic. Both affect mood and "
        "cognition over time, but Huntington's typically begins in midlife and progresses "
        "faster.",

    # ── Parkinson's vs Stroke ─────────────────────────────────────────────────
    ("parkinson", "stroke"):
        "**Parkinson's** is a slowly progressive neurodegenerative disease caused by dopamine "
        "loss, while **stroke** is an acute vascular emergency caused by disrupted blood flow "
        "to the brain. Both can affect movement, but the onset is the key difference: "
        "Parkinson's symptoms develop over months to years; stroke symptoms appear suddenly "
        "in minutes. Some stroke survivors develop *vascular parkinsonism* — a stroke-related "
        "condition that mimics Parkinson's but typically does not respond well to levodopa and "
        "lacks the classic resting tremor.",

    # ── ALS / Huntington's vs Stroke ─────────────────────────────────────────
    ("als", "stroke"):
        "**ALS** and **stroke** can both cause weakness and difficulty speaking or swallowing, "
        "which is why they are sometimes confused — but they are fundamentally different. "
        "Stroke is an *acute vascular event*: a sudden interruption of blood supply to the "
        "brain. Symptoms appear within seconds to minutes and may partially recover. ALS is a "
        "*progressive neurodegenerative disease*: motor neuron loss builds slowly over months "
        "and years with no recovery. A stroke is a medical emergency treatable within hours; "
        "ALS is managed over a lifetime with supportive care and medications like riluzole.",

    ("huntington", "stroke"):
        "**Huntington's** is a genetic neurodegenerative disease causing chorea (involuntary "
        "movements), mood changes, and cognitive decline — symptoms that develop gradually "
        "over years. **Stroke** is an acute vascular event causing sudden neurological "
        "symptoms (weakness, speech loss, confusion) that develop in minutes. They are rarely "
        "confused clinically, but both can affect movement and cognition. Huntington's has a "
        "known genetic cause and can be predicted decades before symptoms; stroke risk is "
        "managed through blood pressure, cholesterol, and lifestyle.",

    # ── Dementia vs ALS / Huntington's ───────────────────────────────────────
    ("als", "dementia"):
        "**Dementia** primarily impairs cognition — memory, language, reasoning. **ALS** "
        "primarily destroys motor neurons, causing progressive muscle weakness and paralysis. "
        "They are usually distinct, but about 10–15% of ALS patients also develop "
        "frontotemporal dementia (FTD-ALS), a combination that affects both movement and "
        "cognition. Dementia is far more prevalent and slower in progression; ALS is rarer "
        "and faster, with most patients surviving 2–5 years after diagnosis.",

    ("huntington", "dementia"):
        "**Huntington's disease** is itself a cause of dementia — it causes cognitive decline "
        "alongside its characteristic involuntary movements (chorea). What makes it distinct "
        "from other dementias is its genetic basis (autosomal dominant inheritance) and the "
        "prominence of motor symptoms and psychiatric changes (depression, irritability, "
        "obsessive behaviours) that often appear before significant cognitive decline. Most "
        "other dementias (Alzheimer's, vascular) are not inherited in a predictable way.",

    # ── Dementia vs Stroke ────────────────────────────────────────────────────
    ("dementia", "stroke"):
        "**Stroke** and **dementia** are closely linked — stroke is one of the leading causes "
        "of dementia, called *vascular dementia*. A single large stroke or multiple small "
        "strokes can damage enough brain tissue to impair memory and thinking. Key differences: "
        "Alzheimer's-type dementia develops gradually with no obvious trigger; stroke-related "
        "cognitive decline tends to start suddenly or in step-wise deteriorations after each "
        "stroke event. Brain MRI can distinguish the two by showing vascular lesions versus "
        "the typical atrophy pattern of Alzheimer's.",
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
    is_prevention = any(p in q for p in [
        "reduce risk", "lower risk", "reduce the risk", "lower the risk",
        "reduce chance", "lower chance", "reduce my risk", "lower my risk",
        "help prevent", "how to prevent", "ways to prevent", "prevent alzheimer",
        "prevent parkinson", "prevent dementia", "prevent als", "prevent stroke",
        "protect against", "protective factor", "factors that help",
        "factors help reduce", "help reduce", "avoid getting", "avoid developing",
        "reduce likelihood", "lower likelihood", "slow progression",
        "delay onset", "lower your risk", "reduce your risk",
    ])
    is_risk = not is_prevention and any(w in q for w in [
        "risk", "cause", "factor", "likely", "prevent", "avoid",
        "predispos", "linked to", "associated with",
    ])
    is_progression = any(w in q for w in [
        "progression", "get worse", "worsen", "prognosis", "long term",
        "survival", "life expectancy", "stage", "advance",
    ])
    # Default to treatment if nothing else matched
    if not any([is_comparison, is_symptom_q, is_diagnosis,
                is_treatment, is_risk, is_prevention, is_progression]):
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
                + _source_block(indexed)
                + closing
            )

    # ── COMPARISON path: clinical blurb first; return early when blurb exists ──
    if is_comparison:
        comparison_blurb = _get_comparison_context(question, diseases)
        if comparison_blurb:
            closing = (
                "\n\n> ⚕️ **Important:** A definitive diagnosis always requires a "
                "qualified clinician. Workup typically includes neurological examination, "
                "blood tests, and brain imaging (MRI or PET)."
            )
            return (
                f"Here's how these conditions differ:\n\n{comparison_blurb}"
                + closing
                + _source_block(indexed)
            )
        # No pre-written blurb — fall through to synthesised per-paper answer
        opening = (
            f"These conditions have distinct characteristics. "
            f"Here's what {n} studies in the literature found:\n"
        )
        intent_key = "comparison"

    # ── TREATMENT path ────────────────────────────────────────────────────────
    elif is_treatment:
        treatment_knowledge = _get_treatment_answer(diseases)
        if treatment_knowledge:
            closing = (
                "\n\n> ⚕️ Treatments vary by stage and individual. "
                "Always consult a qualified clinician before starting or changing any medication."
            )
            return (
                f"Here's an overview of treatments for {disease_str}:\n\n"
                + treatment_knowledge
                + _source_block(indexed)
                + closing
            )
        opening = (
            f"Several approaches have been studied for {disease_str}. "
            f"Here's a synthesis of what the literature shows:\n"
        )
        intent_key = "treatment"

    # ── PREVENTION path ───────────────────────────────────────────────────────
    elif is_prevention:
        prevention_knowledge = _get_prevention_answer(diseases)
        if prevention_knowledge:
            closing = (
                "\n\n> ⚕️ These are population-level strategies based on current evidence. "
                "Individual recommendations should come from a qualified clinician."
            )
            return (
                f"Here are the evidence-based ways to reduce the risk of {disease_str}:\n\n"
                + prevention_knowledge
                + _source_block(indexed)
                + closing
            )
        opening = (
            f"Research has identified several strategies that may reduce the risk of "
            f"{disease_str}. Here's what the evidence shows:\n"
        )
        intent_key = "risk"

    # ── RISK path ─────────────────────────────────────────────────────────────
    elif is_risk:
        risk_knowledge = _get_risk_factor_answer(diseases)
        if risk_knowledge:
            closing = (
                "\n\n> ⚕️ Risk factors indicate population-level associations. "
                "Individual risk should be assessed by a qualified clinician."
            )
            return (
                f"Here are the known risk factors for {disease_str}:\n\n"
                + risk_knowledge
                + _source_block(indexed)
                + closing
            )
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
