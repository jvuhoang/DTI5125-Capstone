# NORA — Neurodegenerative Disease Research Assistant

A conversational chatbot for neurodegenerative and neurological disease research. The system combines biomedical named entity recognition (NER), structured clinical evidence extraction (PICOS), machine learning KNN clustering and 3 methods of classification, and retrieval-augmented generation (RAG) to deliver answers in two layers: a structured clinical summary drawn from curated disease knowledge bases, followed by supporting evidence retrieved from peer-reviewed PubMed abstracts with PICOS-formatted citations.

**Live demo:**  https://group2-neuro-chatbot.streamlit.app

**Supported diseases:** Alzheimer's Disease, Parkinson's Disease, ALS & Huntington's, Dementia & MCI, and Stroke.

---

## How It Works

NORA is built in two stages:

**Offline build (run once):** Downloads ~1200 PubMed abstracts, extracts biomedical entities and clinical evidence structure, trains disease classifiers, and builds a semantic search index. This takes 60–90 minutes and only needs to be done once.

**Runtime (chatbot):** Loads all pre-built models and indexes from disk. Starts in seconds. No re-training, no re-downloading.

When you ask a question, NORA:
1. Identifies disease-related entities in your query (scispaCy NER)
2. Searches the abstract database for the most relevant clinical studies (FAISS semantic search)
3. Formats retrieved evidence using the PICOS framework (Population, Intervention, Comparison, Outcome, Study design)
4. Generates a grounded, cited answer from that evidence (HuggingFace LLM)

---

## Answer Architecture

NORA uses a two-layer answer system to ensure answers are both clinically grounded and supported by peer-reviewed literature.

### Layer 1 — Clinical Knowledge Base

The first layer consists of structured knowledge bases embedded in `rag_pipeline.py`, covering five disease groups: Alzheimer's Disease, Parkinson's Disease, ALS & Huntington's Disease, Dementia & Mild Cognitive Impairment, and Stroke. Each disease group has four dedicated knowledge stores:

| Knowledge base | Content | Used for |
|---|---|---|
| `_DISEASE_SYMPTOMS` | Cardinal symptoms, early signs, and disease-specific presentations | Symptom and "how do I know" questions |
| `_DISEASE_TREATMENTS` | FDA-approved therapies, symptomatic medications, and non-pharmacological approaches | Treatment and management questions |
| `_DISEASE_RISK_FACTORS` | Non-modifiable and modifiable risk factors, with population-level evidence notes | Risk, causes, and prevention questions |
| `_DIAGNOSIS_NOTES` | Clinical diagnostic criteria, standard assessments, and differential diagnosis pointers | Diagnosis and "how is it diagnosed" questions |

When a question is received, the intent routing logic in `_generate_picos_answer()` classifies the query into one of these categories (symptom, treatment, risk, diagnosis, comparison, or progression) and selects the appropriate knowledge base. The response begins with a structured clinical summary drawn from that knowledge base.

### Layer 2 — PubMed RAG Retrieval

After the clinical summary, NORA appends a **Sources** section. This section retrieves the top-*k* most semantically relevant PubMed abstracts from the FAISS index and presents compact, citation-formatted support sentences — one per study. Each citation is formatted as either:

- A clean outcome sentence derived from the abstract's PICOS **Outcome** field, when that field is concise and informative, or
- A bibliographic reference (*Study title*, study design, year) when the outcome field is absent or too technical.


### Why two layers?

The clinical knowledge base provides stable, well-established information (e.g., levodopa as first-line Parkinson's therapy; APOE ε4 as the strongest genetic risk factor for Alzheimer's). The RAG layer adds specificity and recency — surfacing recent trials, intervention comparisons, and population-specific findings from the abstract corpus. Together, the two layers produce answers that are both reliable and evidence-linked.

---

## Requirements

- [Anaconda or Miniconda](https://docs.anaconda.com/miniconda/) installed
- Python 3.11 (handled automatically by `setup.sh`)
- Internet connection for the first run (downloads models and PubMed data)
- ~5 GB free disk space

No API keys are required. All models run locally.

---

## Quick Start

### Step 1 — Clone or download the project

Place all project files in a single folder. Open a terminal and navigate to that folder:

```bash
cd path/to/Project Folder
```

### Step 2 — Run the setup script

This creates a Python 3.11 conda environment called `nora` and installs all dependencies including the biomedical NER model:

```bash
bash setup.sh
```

This takes about 5–10 minutes on a typical internet connection.

### Step 3 — Activate the environment

```bash
conda activate nora
```

You need to do this every time you open a new terminal session.

### Step 4 — Build the pipeline (run once)

```bash
python run_pipeline.py --skip-biobert
```

This runs all offline phases in sequence:

| Phase | What it does | Approx. time |
|-------|-------------|-------------|
| Phase 1 | Downloads ~1200 PubMed abstracts into `abstracts.db` | 5–10 min |
| Phase 2A | Extracts disease and chemical entities (scispaCy NER) | 2–3 min |
| Phase 2B | Extracts PICOS structure from each abstract (bart-large-mnli) | 40–60 min |
| Phase 3 | Trains disease classifiers (LinearSVC, Logistic Regression, BioBERT) | 10-15 min |
| Phase 3b | Builds a biomedical knowledge graph | 1 min |
| Phase 4 | Generates semantic embeddings and builds FAISS search index | 5–10 min |
| Phase 6 | Trains PICOS-based intervention recommender (SVD, KNNBasic, NMF) | 1–2 min |

> **Note on Phase 6:** This phase requires `scikit-surprise`, which must be installed separately (`pip install scikit-surprise`). It is not included in `requirements.txt` because it does not build on Python 3.14 (Streamlit Cloud's environment). Phase 6 is an offline analysis step — the Streamlit app runs without it.

> **Note on `--skip-biobert`:** This skips BioBERT fine-tuning, which can take several hours on a CPU. The chatbot works fully without it using the sklearn classifiers. To include BioBERT (requires a GPU), run without the flag.

If the pipeline is interrupted, re-run it from the phase that failed — it will skip everything already completed:

```bash
python run_pipeline.py --from phase2b
```

### Step 5 — Start the chatbot

```bash
streamlit run streamlit_app.py
```

A browser window opens automatically at `http://localhost:8501`.

---

## File Structure

```
DTI5125 Capstone/
│
├── setup.sh                    # One-time environment setup
├── run_pipeline.py             # Master build script (run before chatbot)
├── requirements.txt            # Python dependencies
│
├── phase1_collect.py           # PubMed data collection
├── phase2_ner_picos.py         # Part A: scispaCy NER
├── phase2b_picos.py            # Part B: PICOS extraction (HuggingFace)
├── phase3_ml.py                # Clustering + disease classification
├── phase3b_knowledge_graph.py  # Knowledge graph construction
├── phase4_rag.py               # Semantic embeddings + FAISS index
├── phase6_recommendersubsys.py # PICOS intervention recommender (offline, local only)
│
├── rag_pipeline.py             # RAG retriever + answer generator (runtime)
├── template_filler.py          # Clinical intake template slot-filling (runtime)
├── conversation_manager.py     # Multi-turn conversation routing (runtime)
├── symptom_scorer.py           # Ensemble disease probability scoring (runtime)
├── streamlit_app.py            # Chatbot frontend
├── webhook.py                  # Flask webhook for Dialogflow integration
│
├── abstracts.db                # Generated: SQLite database of abstracts
├── abstracts.faiss             # Generated: FAISS vector index
├── faiss_id_map.pkl            # Generated: FAISS position → DB row mapping
├── disease_classifier.pkl      # Generated: trained SVM classifier
├── tfidf_vectorizer.pkl        # Generated: fitted TF-IDF vectorizer
├── label_encoder.pkl           # Generated: disease label encoder
├── models/sentence_bert/       # Generated: cached Sentence-BERT model
├── biobert_classifier/         # Generated: fine-tuned BioBERT (if trained)
│
├── silhouette_plot.png         # Generated: cluster evaluation chart
├── cluster_pca_plot.png        # Generated: 2D cluster visualisation
├── knowledge_graph.png         # Generated: biomedical entity co-occurrence graph
├── knowledge_graph.gexf        # Generated: graph export for Gephi
├── knowledge_graph_data.json   # Generated: graph export for D3.js
├── recommender.pkl             # Generated: trained recommender model (Phase 6)
└── recommender_comparison.png  # Generated: SVD vs KNNBasic vs NMF evaluation chart
```

---

## Using the Chatbot

Once running, the chatbot has two main modes:

**Conversational intake:** Describe symptoms and the chatbot asks follow-up questions to build a structured clinical profile, then estimates the most likely disease category with a confidence score.

**Literature Q&A:** Ask research questions about any of the supported diseases. The chatbot first presents a structured clinical summary drawn from its knowledge base, then appends supporting evidence retrieved from the PubMed abstract corpus with compact citations.

**Example questions:**
- "What are the early symptoms of Parkinson's disease?"
- "What interventions have been studied for ALS progression?"
- "What are the outcomes of levodopa therapy?"
- "My patient has resting tremor and rigidity — what could this indicate?"

The **PICOS Literature Explorer** in the sidebar lets you search the abstract database directly and filter results by disease or PICOS element (Population, Intervention, Outcome, etc.).

---

## Running Individual Phases

If you want to run a single phase on its own:

```bash
python phase1_collect.py                    # collect PubMed data
python phase2_ner_picos.py --part a         # NER only
python phase2b_picos.py                     # PICOS extraction only
python phase3_ml.py --skip-biobert          # ML classifiers only
python phase3b_knowledge_graph.py           # knowledge graph only
python phase4_rag.py                        # build FAISS index only
python phase6_recommendersubsys.py          # intervention recommender (requires scikit-surprise)
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'spacy'`**
Run `bash setup.sh` first, then `conda activate nora` before running any scripts.

**`conda: command not found`**
Install [Miniconda](https://docs.anaconda.com/miniconda/) and restart your terminal.

**Phase 2B is slow**
This is normal on CPU. `facebook/bart-large-mnli` processes ~8–10 abstracts per minute on CPU. The script saves progress to the database every 50 abstracts, so it is safe to interrupt and resume.

**`FileNotFoundError: FAISS index not found`**
The pipeline has not been run yet, or Phase 4 did not complete. Run `python run_pipeline.py --from phase4`.

**Chatbot starts but gives no answers**
Check that `abstracts.db`, `abstracts.faiss`, `faiss_id_map.pkl`, `disease_classifier.pkl`, and `tfidf_vectorizer.pkl` all exist in the project folder. If any are missing, re-run the pipeline from the appropriate phase.

**`scikit-surprise` fails to build on Streamlit Cloud**
`scikit-surprise` uses Cython extensions that do not compile on Python 3.14 (the version used by Streamlit Community Cloud). It is intentionally excluded from `requirements.txt`. Phase 6 is an offline pipeline step — install it locally only:
```bash
conda activate nora
pip install scikit-surprise
```

**Port 8501 already in use**
Run on a different port: `streamlit run streamlit_app.py --server.port 8502`

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Data collection | PubMed E-utilities API |
| Database | SQLite |
| Biomedical NER | scispaCy + BC5CDR model |
| PICOS extraction | facebook/bart-large-mnli (zero-shot) |
| Disease classification | LinearSVC, Logistic Regression, BioBERT |
| Semantic search | Sentence-BERT + FAISS |
| Knowledge graph | NetworkX → Gephi / D3.js |
| Recommender system | scikit-surprise (SVD, KNNBasic, NMF) — offline only |
| Answer generation | HuggingFace Transformers |
| Frontend | Streamlit |
| Webhook | Flask |

---

## Notes

- The pipeline must be run before launching the chatbot for the first time.
- All models are saved locally after the first run — no repeated downloads.
- The chatbot does not replace medical advice. All outputs include a disclaimer.
- BioBERT fine-tuning (`--skip-biobert` off) requires a CUDA-capable GPU and significantly increases training time.
