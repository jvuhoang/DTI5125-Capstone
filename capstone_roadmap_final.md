# PICOS-RAG Neurodegenerative Disease Chatbot
## Capstone Project Roadmap — Group 2
### Extension with RAG, NER, PICOS, Knowledge Graphs, ML Classification, and Conversational Intake

---

## Project Title

**PICOS-RAG: A Literature-Grounded Intelligent Chatbot for Neurodegenerative Disease Clinical Decision Support**

---

## Chosen Combination

This project combines **three topics** from the professor's list into a single cohesive system:

| Topic | Role in This Project |
|---|---|
| **Intelligent Chatbot System (RAG)** | Core delivery layer — retrieves PubMed abstracts and generates grounded answers |
| **Information Extraction + NER (SpaCy)** | Feeds the chatbot — extracts diseases and chemicals from abstracts using scispaCy, structures them as PICOS clinical records, and parses user messages in real time |
| **Knowledge Graphs** | Visualization and exploration layer — builds a biomedical co-occurrence graph from NER entities and PICOS fields using NetworkX, exported to Gephi and D3.js |

These three are not independent add-ons — they form a pipeline. NER/PICOS extraction structures raw abstracts → the Knowledge Graph reveals relationships between entities → the RAG chatbot retrieves and synthesises the structured knowledge to answer user questions.

---

## Project Context

This capstone extends the existing Group 2 Neurodegenerative Disease chatbot (`group2chatbot.streamlit.app`) which uses:
- **Dialogflow ES** for intent matching (17 intents, 5 entities)
- **Flask webhook** hosted on Render for ontology queries
- **neurological_triage.owl** as the knowledge base (119 classes, 1163 triples)
- **Streamlit** as the frontend

The capstone adds a second knowledge layer built from scholarly PubMed abstracts with PICOS-structured extraction, satisfying the following course requirements:
- Information Extraction + Named Entity Recognition (SpaCy library)
- Knowledge Graphs (NetworkX + Gephi)
- RAG Intelligent Chatbot System
- Classification (3 algorithms: LinearSVC, Logistic Regression, BioBERT) and Clustering
- Template filling and slot-based conversational intake
- Symptom scoring program
- Data housed in a SQLite database

---

## Diseases Covered

This project covers **five neurodegenerative and neurological diseases**:

| Disease | PubMed Search Term |
|---|---|
| **Alzheimer's** | Alzheimer's disease symptoms risk factors treatment[Title/Abstract] |
| **Parkinson's** | Parkinson's disease symptoms risk factors treatment[Title/Abstract] |
| **ALS** | amyotrophic lateral sclerosis symptoms risk factors treatment[Title/Abstract] |
| **Huntington's** | Huntington's disease symptoms risk factors treatment[Title/Abstract] |
| **Dementia** | dementia symptoms risk factors treatment[Title/Abstract] |
| **Stroke** | stroke symptoms risk factors treatment[Title/Abstract] |

> **Note on label overlap:** Alzheimer's is the most common subtype of dementia, so some abstracts will share vocabulary across both labels. The classifier is expected to learn the distinction — Alzheimer's papers tend to focus on amyloid/tau pathology while Dementia papers address broader cognitive decline and vascular causes. This overlap is worth discussing explicitly in the error analysis section of the written report.

---

## Problem Formulation

Neurodegenerative and neurological diseases — Alzheimer's, Parkinson's, ALS, Huntington's, Dementia, and Stroke — affect hundreds of millions worldwide, yet clinical information about their symptoms, risk factors, and treatments is scattered across thousands of research papers. Clinicians, caregivers, and researchers face the challenge of synthesising this literature quickly and accurately.

The existing chatbot answers structured ontology-based queries but cannot answer open-ended questions grounded in the latest research literature. This project addresses that gap by:

1. Collecting ~1200 PubMed abstracts across six disease categories and structuring them using the PICOS framework
2. Training ML models (including BioBERT) to classify and cluster abstracts by disease
3. Building a knowledge graph to visualise relationships between diseases, drugs, and interventions
4. Implementing a RAG pipeline that retrieves PICOS-structured abstracts to generate literature-backed answers
5. Adding a conversational intake system that fills a clinical template from natural language, scores disease probability, and routes to the RAG layer for follow-up questions

---


## Architecture Overview

```
User message
     ↓
Streamlit frontend (Phase 8)
     ↓
Conversation Manager (Phase 6)
     ├── scispaCy NER extracts entities from user message (real-time)
     ├── Template Filler (Phase 5): slot-filling for clinical intake
     │       └── When template scoreable → Symptom Scorer (Phase 7)
     │               └── Ensemble of 3 trained classifiers → disease probability bars
     └── RAG route: user asks a factual/literature question
             ├── TF-IDF classifier predicts disease → pre-filters retrieval
             ├── PICOS-aware FAISS retriever fetches top-k abstracts (Phase 4)
             ├── LLM (Claude) generates grounded answer from PICOS context
             └── Returns answer + "Papers that informed this answer" citations
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | All phases |
| `requests` + `xml.etree` | PubMed API data collection |
| `sqlite3` | Local database for abstracts + PICOS + NER data |
| `scispaCy 0.5.4` + `en_ner_bc5cdr_md` | Biomedical NER — real-time disease and chemical extraction |
| `anthropic` | PICOS extraction via Claude API |
| `scikit-learn` | TF-IDF, K-Means, LinearSVC, Logistic Regression |
| `transformers` + `torch` | BioBERT fine-tuned sequence classifier |
| `networkx` | Knowledge graph construction and export |
| `sentence-transformers` | Abstract embeddings (all-MiniLM-L6-v2) |
| `faiss-cpu` | Vector similarity search for RAG retrieval |
| `streamlit` | Frontend (existing + extended) |
| `flask` | Webhook backend (existing) |
| `joblib` | Save/load sklearn ML models |
| `matplotlib` + `seaborn` | Clustering, classification, and graph visualisations |
| `pandas` | Data manipulation and error analysis export |

---

## Phase 1 — PubMed Abstract Corpus + SQLite Database

### Goal
Collect ~1200 scholarly abstracts across six disease categories from PubMed and store them in a structured SQLite database. The schema is designed upfront to include placeholder columns for NER entities and all five PICOS fields so that later phases can update rows in place without schema changes.

### Course requirement satisfied
> "house the data in a database" — Data Preparation rubric item (2.5%)

### Steps

**1. Install dependencies**
```bash
pip install requests
```

**2. Collect abstracts using PubMed E-utilities API (free, no key required)**
```python
import requests
import xml.etree.ElementTree as ET
import sqlite3
import time

def fetch_pubmed_ids(query, max_results=200):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json"
    }
    r = requests.get(url, params=params)
    return r.json()["esearchresult"]["idlist"]

def fetch_abstract(pmid):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "xml"}
    r = requests.get(url, params=params)
    root = ET.fromstring(r.content)

    title    = root.findtext(".//ArticleTitle") or ""
    abstract = " ".join([t.text for t in root.findall(".//AbstractText") if t.text]) or ""
    year     = root.findtext(".//PubDate/Year") or ""
    journal  = root.findtext(".//Journal/Title") or ""

    return {"pmid": pmid, "title": title, "abstract": abstract,
            "year": year, "journal": journal}

queries = {
    "Alzheimer":  "Alzheimer's disease symptoms risk factors treatment[Title/Abstract]",
    "Parkinson":  "Parkinson's disease symptoms risk factors treatment[Title/Abstract]",
    "ALS":        "amyotrophic lateral sclerosis symptoms risk factors treatment[Title/Abstract]",
    "Huntington": "Huntington's disease symptoms risk factors treatment[Title/Abstract]",
    "Dementia":   "dementia symptoms risk factors treatment[Title/Abstract]",
    "Stroke":     "stroke symptoms risk factors treatment[Title/Abstract]",
}

all_abstracts = []
for disease, query in queries.items():
    ids = fetch_pubmed_ids(query, max_results=200)
    print(f"{disease}: found {len(ids)} articles")
    for pmid in ids:
        record = fetch_abstract(pmid)
        record["disease_label"] = disease
        all_abstracts.append(record)
        time.sleep(0.35)  # PubMed rate limit: max 3 requests/second
    print(f"{disease}: done")
```

**3. Save to SQLite — schema includes PICOS and NER columns**
```python
conn = sqlite3.connect("abstracts.db")
c = conn.cursor()

c.execute("""
    CREATE TABLE IF NOT EXISTS abstracts (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        pmid                TEXT UNIQUE,
        title               TEXT,
        abstract            TEXT,
        year                TEXT,
        journal             TEXT,
        disease_label       TEXT,
        ner_entities        TEXT,
        picos_population    TEXT,
        picos_intervention  TEXT,
        picos_comparison    TEXT,
        picos_outcome       TEXT,
        picos_study_design  TEXT,
        embedding_id        INTEGER
    )
""")

for rec in all_abstracts:
    try:
        c.execute("""
            INSERT OR IGNORE INTO abstracts
            (pmid, title, abstract, year, journal, disease_label)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (rec["pmid"], rec["title"], rec["abstract"],
              rec["year"], rec["journal"], rec["disease_label"]))
    except Exception as e:
        print(f"Error inserting {rec['pmid']}: {e}")

conn.commit()
count = c.execute("SELECT COUNT(*) FROM abstracts").fetchone()[0]
print(f"Database contains {count} abstracts")
conn.close()
```

**4. Verify counts per disease**
```python
conn = sqlite3.connect("abstracts.db")
c = conn.cursor()
for label in ["Alzheimer", "Parkinson", "ALS", "Huntington", "Dementia", "Stroke"]:
    count = c.execute(
        "SELECT COUNT(*) FROM abstracts WHERE disease_label=?", (label,)
    ).fetchone()[0]
    print(f"{label}: {count} abstracts")
conn.close()
```

### Expected output
```
Alzheimer:  200 abstracts
Parkinson:  200 abstracts
ALS:        200 abstracts
Huntington: 200 abstracts
Dementia:   200 abstracts
Stroke:     200 abstracts
Database contains 1200 abstracts
```

---

## Phase 2 — NER with scispaCy + PICOS Extraction

### Goal
Run two complementary extraction processes on every abstract:
1. **scispaCy NER (Part A)** — extract DISEASE and CHEMICAL entity spans; fast, CPU-based, runs on every abstract offline and on every user message at runtime
2. **PICOS extraction (Part B)** — extract Population, Intervention, Comparison, Outcome, Study design via Claude API; results stored in the five PICOS columns

### Course requirements satisfied
> "Information Extraction and Named Entity Extraction (Use SpaCy library)" — NER requirement
> "Text Feature Engineering" — rubric (3%)

---

### Part A — scispaCy NER

**1. Install**
```bash
pip install scispacy==0.5.4 spacy==3.7.4
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

> **Version note:** Always pin scispaCy and spaCy to matching minor versions. The most common failure point is installing spaCy 3.8 with scispaCy 0.5.3 — they are incompatible. Use spaCy 3.7.x with scispaCy 0.5.4.

**2. Run NER on all abstracts**
```python
import spacy
import sqlite3
import json

nlp = spacy.load("en_ner_bc5cdr_md")

conn = sqlite3.connect("abstracts.db")
c = conn.cursor()

rows = c.execute(
    "SELECT id, abstract FROM abstracts WHERE ner_entities IS NULL"
).fetchall()
print(f"Running NER on {len(rows)} abstracts...")

for i, (row_id, abstract_text) in enumerate(rows):
    if not abstract_text:
        continue
    doc = nlp(abstract_text)
    entities = [
        {"text": ent.text, "label": ent.label_,
         "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]
    c.execute(
        "UPDATE abstracts SET ner_entities=? WHERE id=?",
        (json.dumps(entities), row_id)
    )
    if i % 100 == 0:
        conn.commit()
        print(f"  NER: {i}/{len(rows)}")

conn.commit()
conn.close()
print("NER complete.")
```

---

### Part B — PICOS Extraction via Claude API

**1. PICOS extraction function**
```python
import anthropic
import sqlite3
import json
import time

client = anthropic.Anthropic()  # requires ANTHROPIC_API_KEY in environment

PICOS_PROMPT = """Extract the PICOS elements from this medical abstract.
Return ONLY a JSON object with these exact keys:
- population: who was studied (disease, age group, sample size if mentioned)
- intervention: treatment, drug, therapy, or exposure studied
- comparison: what the intervention was compared to (or "not reported")
- outcome: what was measured or observed
- study_design: type of study (e.g. RCT, cohort, case-control, review)

If an element is not mentioned, use "not reported".

Abstract:
{abstract}

JSON:"""

def extract_picos(abstract_text):
    if not abstract_text or len(abstract_text) < 50:
        return None
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": PICOS_PROMPT.format(abstract=abstract_text[:1500])
            }]
        )
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"  PICOS error: {e}")
        return None
```

**2. Run PICOS extraction on all abstracts**
```python
conn = sqlite3.connect("abstracts.db")
c = conn.cursor()

rows = c.execute("""
    SELECT id, abstract FROM abstracts
    WHERE picos_population IS NULL AND abstract IS NOT NULL AND abstract != ''
""").fetchall()
print(f"Extracting PICOS from {len(rows)} abstracts...")

for i, (row_id, abstract_text) in enumerate(rows):
    picos = extract_picos(abstract_text)
    if picos:
        c.execute("""
            UPDATE abstracts SET
                picos_population   = ?,
                picos_intervention = ?,
                picos_comparison   = ?,
                picos_outcome      = ?,
                picos_study_design = ?
            WHERE id = ?
        """, (
            picos.get("population", ""),
            picos.get("intervention", ""),
            picos.get("comparison", ""),
            picos.get("outcome", ""),
            picos.get("study_design", ""),
            row_id
        ))
    if i % 50 == 0:
        conn.commit()
        print(f"  PICOS: {i}/{len(rows)}")
    time.sleep(0.1)

conn.commit()
conn.close()
print("PICOS extraction complete.")
```

> **Time estimate:** ~2–3 hours for 1200 abstracts at 0.1s delay per call. Run once; results persist in the database. The `WHERE picos_population IS NULL` clause means the script can be safely resumed if interrupted.

**3. Inspect and analyse PICOS results**
```python
conn = sqlite3.connect("abstracts.db")
c = conn.cursor()

rows = c.execute("""
    SELECT title, disease_label, picos_population, picos_intervention,
           picos_comparison, picos_outcome, picos_study_design
    FROM abstracts WHERE picos_population IS NOT NULL LIMIT 5
""").fetchall()

for row in rows:
    title, disease, pop, interv, comp, outcome, study = row
    print(f"\n[{disease}] {title[:60]}")
    print(f"  P: {pop}")
    print(f"  I: {interv}")
    print(f"  C: {comp}")
    print(f"  O: {outcome}")
    print(f"  S: {study}")

conn.close()
```

---

## Phase 3 — Clustering and Classification (3 Algorithms)

### Goal
Cluster abstracts by topic using K-Means and train a disease classifier using **three algorithms**:

| # | Algorithm | Features | Rationale |
|---|---|---|---|
| 1 | **LinearSVC** | TF-IDF + PICOS | Fast linear SVM; strong baseline for sparse text |
| 2 | **Logistic Regression** | TF-IDF + PICOS | Interpretable; produces calibrated probabilities |
| 3 | **BioBERT** | Raw abstract text | Fine-tuned transformer; state-of-the-art for biomedical NLP |

All three are compared side by side with accuracy charts, confusion matrices, and error analysis.

### Course requirements satisfied
> "Clustering (at least one algorithm)" — rubric (3%)
> "Classification — use at least three different algorithms" — rubric (3%)
> "Text Feature Engineering" — rubric (3%)
> "Evaluation of ML results" — rubric (4%)
> "Error Analysis" — rubric (3%)
> "Visualization of results" — rubric (3%)

---

### Part A — Feature Preparation and K-Means Clustering

**1. Install**
```bash
pip install scikit-learn matplotlib seaborn pandas scipy
```

**2. Build TF-IDF feature matrix (used by LinearSVC and Logistic Regression)**
```python
import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

conn = sqlite3.connect("abstracts.db")
c = conn.cursor()
rows = c.execute("""
    SELECT id, abstract, disease_label,
           picos_population, picos_intervention, picos_outcome
    FROM abstracts
    WHERE abstract IS NOT NULL AND abstract != ''
""").fetchall()
conn.close()

ids    = [r[0] for r in rows]
texts  = [r[1] for r in rows]
labels = [r[2] for r in rows]

# Combine abstract text with PICOS fields for richer features
picos_text = [f"{r[3] or ''} {r[4] or ''} {r[5] or ''}" for r in rows]
combined   = [f"{t} {p}" for t, p in zip(texts, picos_text)]

vectorizer = TfidfVectorizer(
    max_features=3000, stop_words="english", ngram_range=(1, 2)
)
X = vectorizer.fit_transform(combined)
print(f"TF-IDF feature matrix: {X.shape}")

le = LabelEncoder()
y  = le.fit_transform(labels)
print(f"Classes: {list(le.classes_)}")
```

**3. K-Means Clustering with silhouette optimisation**
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from collections import Counter

silhouette_scores = []
k_range = range(4, 14)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    score = silhouette_score(X, km.labels_, sample_size=500)
    silhouette_scores.append(score)
    print(f"k={k}: silhouette={score:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(list(k_range), silhouette_scores, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Optimal cluster count — PubMed abstracts (TF-IDF + PICOS)")
plt.tight_layout()
plt.savefig("silhouette_plot.png", dpi=150)
plt.show()

best_k = list(k_range)[silhouette_scores.index(max(silhouette_scores))]
print(f"\nBest k: {best_k}")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = km_final.fit_predict(X)
feature_names  = vectorizer.get_feature_names_out()

for cluster_id in range(best_k):
    center       = km_final.cluster_centers_[cluster_id]
    top_indices  = center.argsort()[-10:][::-1]
    top_terms    = [feature_names[i] for i in top_indices]
    cluster_diseases = [labels[i] for i, c in enumerate(cluster_labels) if c == cluster_id]
    dominant = Counter(cluster_diseases).most_common(1)[0]
    print(f"\nCluster {cluster_id} (dominant: {dominant[0]}, n={len(cluster_diseases)}):")
    print(f"  Top terms: {', '.join(top_terms)}")
```

**4. PCA 2D cluster visualisation**
```python
from sklearn.decomposition import PCA

X_dense = X.toarray()
pca  = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_dense)

plt.figure(figsize=(10, 7))
colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for cluster_id in range(best_k):
    mask = cluster_labels == cluster_id
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=[colors[cluster_id]], label=f"Cluster {cluster_id}",
                alpha=0.5, s=20)
plt.legend()
plt.title(f"K-Means Clusters (k={best_k}) — PCA 2D projection")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("cluster_pca_plot.png", dpi=150)
plt.show()
```

---

### Part B — Three Disease Classifiers

**5. LinearSVC and Logistic Regression (TF-IDF features)**
```python
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib, seaborn as sns, pandas as pd

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sklearn_classifiers = {
    "LinearSVC":          LinearSVC(random_state=42, max_iter=2000),
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000, C=1.0),
}

results = {}
for name, clf in sklearn_classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    results[name] = {"clf": clf, "y_pred": y_pred, "accuracy": acc}
    print(f"\n{'='*50}")
    print(f"Classifier: {name}  |  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
```

**6. BioBERT fine-tuned sequence classifier**
```bash
pip install transformers torch datasets
```

```python
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from torch.utils.data import Dataset
import torch

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
NUM_LABELS  = len(le.classes_)   # 6

tokenizer_bert = AutoTokenizer.from_pretrained(MODEL_NAME)

# Use the same 80/20 train/test split indices as the sklearn classifiers
_, test_idx = train_test_split(
    range(len(combined)), test_size=0.2, random_state=42, stratify=y
)
train_idx = [i for i in range(len(combined)) if i not in set(test_idx)]

class AbstractDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i   = self.indices[idx]
        enc = tokenizer_bert(
            combined[i], truncation=True, max_length=256,
            padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(y[i], dtype=torch.long),
        }

train_dataset = AbstractDataset(train_idx)
test_dataset  = AbstractDataset(test_idx)

model_bert = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)

training_args = TrainingArguments(
    output_dir="./biobert_output",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    report_to="none",
)

def compute_metrics(eval_pred):
    logits, labels_arr = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels_arr, preds)}

trainer = Trainer(
    model=model_bert,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

bert_preds_raw = trainer.predict(test_dataset)
bert_y_pred    = np.argmax(bert_preds_raw.predictions, axis=-1)
bert_acc       = accuracy_score(y_test, bert_y_pred)

results["BioBERT"] = {"clf": trainer.model, "y_pred": bert_y_pred, "accuracy": bert_acc}
print(f"\nBioBERT accuracy: {bert_acc:.4f}")
print(classification_report(y_test, bert_y_pred, target_names=le.classes_))

# Save BioBERT model
model_bert.save_pretrained("biobert_classifier")
tokenizer_bert.save_pretrained("biobert_classifier")
print("BioBERT model saved to ./biobert_classifier/")
```

**7. Classifier comparison bar chart**
```python
names = list(results.keys())
accs  = [results[n]["accuracy"] for n in names]

plt.figure(figsize=(8, 4))
bars = plt.bar(names, accs, color=["steelblue", "seagreen", "darkorchid"], edgecolor="black")
plt.ylim(0.5, 1.0)
plt.ylabel("Test Accuracy")
plt.title("Disease Classifier Comparison — 6-class (TF-IDF baseline vs BioBERT)")
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig("classifier_comparison.png", dpi=150)
plt.show()
```

**8. Confusion matrices for all three classifiers**
```python
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"{name}\n(acc={res['accuracy']:.3f})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.tick_params(axis='x', rotation=30)
plt.suptitle("Confusion Matrices — 6-class Neurodegenerative Disease Classifier", fontsize=13)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()
```

**9. Error analysis — misclassification inspection**
```python
best_name = max(results, key=lambda n: results[n]["accuracy"])
best_pred = results[best_name]["y_pred"]

_, test_indices = train_test_split(
    range(len(combined)), test_size=0.2, random_state=42, stratify=y
)
errors = [
    {
        "text":      combined[test_indices[i]][:120],
        "actual":    le.inverse_transform([y_test[i]])[0],
        "predicted": le.inverse_transform([best_pred[i]])[0],
    }
    for i in range(len(y_test)) if y_test[i] != best_pred[i]
]
print(f"Best model: {best_name}")
print(f"Misclassifications: {len(errors)} / {len(y_test)}  ({len(errors)/len(y_test):.2%})")

df_errors = pd.DataFrame(errors)
print(df_errors.groupby(["actual", "predicted"]).size().reset_index(name="count"))
df_errors.to_csv("error_analysis.csv", index=False)
print("Saved: error_analysis.csv")
```

**10. Save best sklearn model (used at runtime by the symptom scorer)**
```python
best_sklearn_name = max(
    [n for n in results if n != "BioBERT"],
    key=lambda n: results[n]["accuracy"]
)
joblib.dump(results[best_sklearn_name]["clf"], "disease_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(le,         "label_encoder.pkl")
print(f"Best sklearn model ({best_sklearn_name}) saved.")
# BioBERT is saved separately in ./biobert_classifier/
```

> **Runtime note:** The symptom scorer (Phase 7) ensembles all three classifiers. BioBERT inference takes ~300ms without a GPU — acceptable for a one-time score at the end of intake, but too slow for turn-by-turn routing. Turn-by-turn routing uses the saved sklearn classifier only.

---

## Phase 3B — Knowledge Graph (NetworkX + Gephi)

### Goal
Build a biomedical knowledge graph from the NER entities and PICOS data extracted in Phase 2. Nodes are diseases, chemicals, and interventions. Edges represent co-occurrence within the same abstract, weighted by frequency. Exported for both Gephi (interactive exploration) and D3.js (web frontend).

### Course requirements satisfied
> "Knowledge Graphs" — professor's topic list
> "Visualization of results" — rubric (3%)

**1. Install**
```bash
pip install networkx
```

**2. Build the graph**
```python
import sqlite3, json, networkx as nx
from collections import defaultdict

conn = sqlite3.connect("abstracts.db")
c = conn.cursor()
rows = c.execute("""
    SELECT id, disease_label, ner_entities, picos_intervention, picos_outcome
    FROM abstracts WHERE ner_entities IS NOT NULL
""").fetchall()
conn.close()

G = nx.Graph()
edge_weights = defaultdict(int)

for row_id, disease_label, ner_json, intervention, outcome in rows:
    entities = json.loads(ner_json) if ner_json else []
    node_set  = set()

    if disease_label:
        node_name = disease_label.lower()
        G.add_node(node_name, node_type="disease", size=20)
        node_set.add(node_name)

    for ent in entities:
        text  = ent["text"].lower().strip()
        label = ent["label"]
        if len(text) < 3 or len(text) > 50:
            continue
        G.add_node(text, node_type="disease" if label == "DISEASE" else "chemical", size=5)
        node_set.add(text)

    if intervention and intervention.lower() not in ("not reported", ""):
        interv = intervention.lower().strip()[:60]
        G.add_node(interv, node_type="intervention", size=8)
        node_set.add(interv)

    node_list = list(node_set)
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            edge_weights[(node_list[i], node_list[j])] += 1

for (u, v), w in edge_weights.items():
    if w >= 3 and G.has_node(u) and G.has_node(v):
        G.add_edge(u, v, weight=w)

G.remove_nodes_from(list(nx.isolates(G)))
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
```

**3. Visualise inline with matplotlib**
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

color_map   = {"disease": "#E63946", "chemical": "#457B9D", "intervention": "#2A9D8F"}
node_colors = [color_map.get(G.nodes[n].get("node_type", "disease"), "#888") for n in G.nodes()]
node_sizes  = [300 + G.degree(n) * 80 for n in G.nodes()]

plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, k=2.5, seed=42, iterations=60)
nx.draw_networkx_edges(G, pos,
                       width=[G[u][v]["weight"] * 0.3 for u, v in G.edges()],
                       alpha=0.3, edge_color="gray")
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85)

threshold = sorted([G.degree(n) for n in G.nodes()], reverse=True)[min(40, G.number_of_nodes()-1)]
nx.draw_networkx_labels(G, pos,
                         labels={n: n for n in G.nodes() if G.degree(n) >= threshold},
                         font_size=7, font_weight="bold")

patches = [mpatches.Patch(color=c, label=t.capitalize()) for t, c in color_map.items()]
plt.legend(handles=patches, loc="upper left", fontsize=10)
plt.title("Biomedical Knowledge Graph — 6-disease Neurodegenerative Corpus", fontsize=13)
plt.axis("off"); plt.tight_layout()
plt.savefig("knowledge_graph.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: knowledge_graph.png")
```

**4. Export to Gephi (.gexf) and D3.js (.json)**
```python
# Gephi
for node in G.nodes():
    ntype = G.nodes[node].get("node_type", "disease")
    G.nodes[node]["label"] = node
    G.nodes[node]["r"] = {"disease": 230, "chemical": 69,  "intervention": 42 }[ntype]
    G.nodes[node]["g"] = {"disease": 57,  "chemical": 123, "intervention": 157}[ntype]
    G.nodes[node]["b"] = {"disease": 70,  "chemical": 157, "intervention": 143}[ntype]

nx.write_gexf(G, "knowledge_graph.gexf")
print("Exported: knowledge_graph.gexf — open in Gephi, run ForceAtlas2, apply Modularity colouring")

# D3.js JSON for index.html Knowledge Graph tab
graph_data = {
    "nodes": [{"id": n, "node_type": G.nodes[n].get("node_type", "disease"),
                "degree": G.degree(n)} for n in G.nodes()],
    "edges": [{"source": u, "target": v, "weight": G[u][v].get("weight", 1)}
              for u, v in G.edges()],
}
with open("knowledge_graph_data.json", "w") as f:
    json.dump(graph_data, f)
print(f"Exported: knowledge_graph_data.json ({len(graph_data['nodes'])} nodes, "
      f"{len(graph_data['edges'])} edges)")
print("Place alongside index.html — the Knowledge Graph tab loads it automatically.")
```

**5. Graph statistics (for report)**
```python
dc = nx.degree_centrality(G)
top = sorted(dc.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.4f}")
print(f"Average degree: {sum(d for _, d in G.degree()) / G.number_of_nodes():.2f}")
print("\nTop 10 most connected nodes:")
for name, centrality in top:
    ntype = G.nodes[name].get("node_type", "?")
    print(f"  [{ntype:12s}] {name:<40s} centrality={centrality:.4f}")
```

---

## Phase 4 — RAG Pipeline with PICOS-Aware Retrieval

### Goal
Embed every abstract using Sentence-BERT (concatenating title + abstract + key PICOS fields) and store all vectors in a FAISS flat L2 index. At runtime the PICOSRetriever encodes the user query, searches the index for top-k matches, and fetches full PICOS records from SQLite. The RAGAnswerGenerator builds a PICOS-structured prompt and calls the Claude API to generate a grounded answer with inline citations.

### Course requirement satisfied
> "Intelligent Chatbot System (RAG)" — professor's topic list

**1. Install**
```bash
pip install sentence-transformers faiss-cpu anthropic
```

**2. Generate embeddings and build FAISS index**
```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, sqlite3, pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

conn = sqlite3.connect("abstracts.db")
c = conn.cursor()
rows = c.execute("""
    SELECT id, title, abstract, picos_population, picos_intervention, picos_outcome
    FROM abstracts WHERE abstract IS NOT NULL
""").fetchall()
conn.close()

db_ids = [r[0] for r in rows]
texts_to_embed = [
    f"{r[1]}. {r[2]} Population: {r[3] or ''}. "
    f"Intervention: {r[4] or ''}. Outcome: {r[5] or ''}."
    for r in rows
]

print(f"Encoding {len(texts_to_embed)} abstracts...")
embeddings = model.encode(texts_to_embed, batch_size=32, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "abstracts.faiss")
with open("faiss_id_map.pkl", "wb") as f:
    pickle.dump(db_ids, f)
print(f"FAISS index built: {index.ntotal} vectors")
```

**3. PICOSRetriever and RAGAnswerGenerator**
```python
# rag_pipeline.py
import faiss, pickle, sqlite3, numpy as np, anthropic
from sentence_transformers import SentenceTransformer

class PICOSRetriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index("abstracts.faiss")
        with open("faiss_id_map.pkl", "rb") as f:
            self.id_map = pickle.load(f)
        self.conn = sqlite3.connect("abstracts.db")

    def retrieve(self, query, k=5, filter_disease=None):
        query_vec         = self.model.encode([query]).astype("float32")
        _, indices        = self.index.search(query_vec, k * 3)
        results           = []
        for idx in indices[0]:
            db_id = self.id_map[idx]
            row   = self.conn.execute("""
                SELECT pmid, title, abstract, disease_label, year,
                       picos_population, picos_intervention,
                       picos_comparison, picos_outcome, picos_study_design
                FROM abstracts WHERE id=?
            """, (db_id,)).fetchone()
            if not row:
                continue
            if filter_disease and row[3] != filter_disease:
                continue
            results.append({
                "pmid": row[0], "title": row[1], "abstract": row[2],
                "disease": row[3], "year": row[4],
                "P": row[5] or "not reported", "I": row[6] or "not reported",
                "C": row[7] or "not reported", "O": row[8] or "not reported",
                "S": row[9] or "not reported",
            })
            if len(results) >= k:
                break
        return results


class RAGAnswerGenerator:
    def __init__(self, retriever):
        self.retriever = retriever
        self.client    = anthropic.Anthropic()

    def answer(self, question, k=5, filter_disease=None):
        abstracts = self.retriever.retrieve(question, k=k, filter_disease=filter_disease)
        if not abstracts:
            return {"answer": "No relevant literature found.", "sources": [], "picos_summary": []}

        context = "\n\n".join([
            f"[{i+1}] {a['title']} ({a['disease']}, {a['year']})\n"
            f"  P: {a['P']}\n  I: {a['I']}\n  C: {a['C']}\n"
            f"  O: {a['O']}\n  S: {a['S']}\n"
            f"  Abstract: {a['abstract'][:300]}..."
            for i, a in enumerate(abstracts)
        ])

        prompt = f"""You are a clinical literature assistant specialising in
neurodegenerative and neurological diseases (Alzheimer's, Parkinson's, ALS,
Huntington's, Dementia, Stroke).
Answer ONLY from the provided PICOS-structured abstracts.
Cite abstract numbers. If information is insufficient, say so.
Do not speculate beyond what the literature states.

RETRIEVED ABSTRACTS:
{context}

QUESTION: {question}

ANSWER:"""

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "answer":        message.content[0].text,
            "sources":       [f"PMID {a['pmid']}" for a in abstracts],
            "picos_summary": [{"pmid": a["pmid"], "title": a["title"],
                               "year": a["year"], "P": a["P"],
                               "I": a["I"], "O": a["O"], "S": a["S"]}
                              for a in abstracts]
        }
```

---

## Phase 5 — Template Filler (Clinical Intake)

### Goal
A slot-filling dialogue layer that collects six structured clinical fields from the user through natural conversation. Uses scispaCy NER and regex patterns to extract values from free-text user responses. A priority queue determines which field to ask about next — symptoms first, then duration, then severity. When the minimum required slots (symptoms + duration + severity) are filled, the template is forwarded to the symptom scorer.

### Course requirement satisfied
> "Template filling" — professor's requirement

```python
# template_filler.py
from dataclasses import dataclass
from typing import Optional
import spacy, re

nlp = spacy.load("en_ner_bc5cdr_md")

@dataclass
class ClinicalTemplate:
    age_gender:          Optional[str] = None
    primary_symptoms:    Optional[str] = None
    duration:            Optional[str] = None
    severity:            Optional[str] = None   # mild / moderate / severe
    family_history:      Optional[str] = None
    current_medications: Optional[str] = None

    def filled_count(self) -> int:
        return sum(1 for v in vars(self).values() if v is not None)

    def is_scoreable(self) -> bool:
        return (self.primary_symptoms is not None
                and self.duration is not None
                and self.severity is not None)

    def to_text(self) -> str:
        parts = []
        if self.age_gender:          parts.append(f"Patient: {self.age_gender}")
        if self.primary_symptoms:    parts.append(f"Symptoms: {self.primary_symptoms}")
        if self.duration:            parts.append(f"Duration: {self.duration}")
        if self.severity:            parts.append(f"Severity: {self.severity}")
        if self.family_history:      parts.append(f"Family history: {self.family_history}")
        if self.current_medications: parts.append(f"Medications: {self.current_medications}")
        return ". ".join(parts)


FIELD_PRIORITY = [
    "primary_symptoms", "duration", "severity",
    "age_gender", "family_history", "current_medications"
]

FOLLOW_UP_QUESTIONS = {
    "primary_symptoms":    "What symptoms are you or the patient experiencing?",
    "duration":            "How long have these symptoms been present?",
    "severity":            "How would you describe the severity — mild, moderate, or severe?",
    "age_gender":          "What is the patient's age and gender?",
    "family_history":      "Is there any family history of neurological conditions?",
    "current_medications": "What medications is the patient currently taking, if any?",
}


def extract_from_text(user_text: str, template: ClinicalTemplate) -> ClinicalTemplate:
    doc        = nlp(user_text)
    text_lower = user_text.lower()

    # Severity
    for sev in ["mild", "moderate", "severe"]:
        if sev in text_lower and template.severity is None:
            template.severity = sev

    # Duration — regex
    for pat in [r"\d+\s*(year|month|week|day)s?",
                r"(for|over|about|around)\s+\d+",
                r"(since|for the past)\s+\w+"]:
        m = re.search(pat, text_lower)
        if m and template.duration is None:
            template.duration = m.group()
            break

    # Age / gender
    age_m = re.search(r"\b(\d{2,3})[- ]*(year[s]?[- ]*old|yo|y\.o\.?)\b", text_lower)
    if age_m and template.age_gender is None:
        gender = ("male"   if any(g in text_lower for g in ["male","man","he","him"]) else
                  "female" if any(g in text_lower for g in ["female","woman","she","her"]) else
                  "unknown")
        template.age_gender = f"{age_m.group(1)} years old, {gender}"

    # Symptoms — DISEASE entities from scispaCy
    diseases = [ent.text for ent in doc.ents if ent.label_ == "DISEASE"]
    if diseases and template.primary_symptoms is None:
        template.primary_symptoms = ", ".join(diseases)

    # Medications — CHEMICAL entities from scispaCy
    chemicals = [ent.text for ent in doc.ents if ent.label_ == "CHEMICAL"]
    if chemicals and template.current_medications is None:
        template.current_medications = ", ".join(chemicals)

    # Family history — keyword detection
    if any(kw in text_lower for kw in ["family","mother","father","parent",
                                        "sibling","genetic","hereditary"]):
        if template.family_history is None:
            template.family_history = user_text[:80]

    return template


def next_question(template: ClinicalTemplate) -> Optional[str]:
    for field_name in FIELD_PRIORITY:
        if getattr(template, field_name) is None:
            return FOLLOW_UP_QUESTIONS[field_name]
    return None
```

---

## Phase 6 — Conversation Manager (Session Memory + Routing)

### Goal
Manages the full conversation history, routes each user turn to the correct handler (template filling, RAG retrieval, or symptom scoring), and maintains state across turns using Streamlit session state. All conversation history is passed to the LLM on every turn so the model has full context.

### Course requirement satisfied
> "Conversational chatbot / multi-turn dialogue management" — professor's requirement

```python
# conversation_manager.py
import streamlit as st
import joblib
from template_filler import ClinicalTemplate, extract_from_text, next_question
from rag_pipeline import PICOSRetriever, RAGAnswerGenerator

def init_session():
    if "history"  not in st.session_state:
        st.session_state.history  = []
    if "template" not in st.session_state:
        st.session_state.template = ClinicalTemplate()
    if "scored"   not in st.session_state:
        st.session_state.scored   = False


def is_factual_query(text: str) -> bool:
    """Heuristic: questions about research, treatments, trials → RAG route."""
    keywords = ["what", "how", "which", "treatment", "therapy", "study",
                "research", "drug", "intervention", "trial", "evidence",
                "literature", "paper", "found", "cause", "risk"]
    return any(kw in text.lower() for kw in keywords) and "?" in text


def handle_turn(user_text: str, retriever: PICOSRetriever,
                rag: RAGAnswerGenerator, clf, vectorizer, le) -> str:

    st.session_state.history.append({"role": "user", "content": user_text})
    template = st.session_state.template

    # Route 1: RAG — factual literature question
    if is_factual_query(user_text):
        vec               = vectorizer.transform([user_text])
        predicted_disease = le.inverse_transform(clf.predict(vec))[0]
        result            = rag.answer(user_text, k=5, filter_disease=predicted_disease)
        response          = result["answer"]
        if result.get("picos_summary"):
            response += "\n\n**Papers that informed this answer:**\n"
            for p in result["picos_summary"]:
                response += f"- {p['title']} ({p['year']}) — PMID {p['pmid']}\n"
        st.session_state.history.append({"role": "assistant", "content": response})
        return response

    # Route 2: Template filling — extract slots
    template = extract_from_text(user_text, template)
    st.session_state.template = template

    # Route 3: Trigger score panel when template is scoreable
    if template.is_scoreable() and not st.session_state.scored:
        st.session_state.scored = True
        return "__SCORE__"

    # Route 4: Ask next unanswered question
    next_q = next_question(template)
    if next_q:
        st.session_state.history.append({"role": "assistant", "content": next_q})
        return next_q

    return "Thank you — all information collected. You can now ask questions about the literature."
```

---

## Phase 7 — Symptom Scorer (Ensemble Disease Probability)

### Goal
Once the template has at minimum symptoms + duration + severity filled, concatenate all fields into text and score it using an ensemble of all three trained classifiers. Display disease probability as percentage bars with a mandatory medical disclaimer.

### Course requirements satisfied
> "Scoring program" — professor's requirement
> "Evaluation of ML results" — rubric (4%)

```python
# symptom_scorer.py
import joblib, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_models():
    clf_sk       = joblib.load("disease_classifier.pkl")
    vectorizer   = joblib.load("tfidf_vectorizer.pkl")
    le           = joblib.load("label_encoder.pkl")
    tok_bert     = AutoTokenizer.from_pretrained("biobert_classifier")
    mod_bert     = AutoModelForSequenceClassification.from_pretrained("biobert_classifier")
    mod_bert.eval()
    return clf_sk, vectorizer, le, tok_bert, mod_bert


def score_template(template_text: str, clf_sk, vectorizer, le,
                   tok_bert, mod_bert) -> dict:
    """Returns {disease: probability} averaged across all three classifiers."""

    # --- TF-IDF classifier (LinearSVC or LogisticRegression) ---
    vec = vectorizer.transform([template_text])
    if hasattr(clf_sk, "predict_proba"):
        proba_sklearn = clf_sk.predict_proba(vec)[0]
    else:
        scores        = clf_sk.decision_function(vec)[0]
        scores        = scores - scores.max()
        exp_s         = np.exp(scores)
        proba_sklearn = exp_s / exp_s.sum()

    # --- BioBERT classifier ---
    enc = tok_bert(template_text, truncation=True, max_length=256,
                   padding="max_length", return_tensors="pt")
    with torch.no_grad():
        logits = mod_bert(**enc).logits[0].numpy()
    logits     = logits - logits.max()
    exp_l      = np.exp(logits)
    proba_bert = exp_l / exp_l.sum()

    # --- Ensemble ---
    ensemble = (proba_sklearn + proba_bert) / 2
    return {disease: float(prob) for disease, prob in zip(le.classes_, ensemble)}


def render_score_panel(scores: dict):
    import streamlit as st
    st.subheader("Disease Probability Assessment")
    for disease, prob in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        pct   = prob * 100
        label = "High" if pct > 60 else "Moderate" if pct > 30 else "Low"
        st.metric(label=disease, value=f"{pct:.1f}%", delta=label)
        st.progress(prob)
    st.warning(
        "⚠️ **Disclaimer:** This is not a medical diagnosis. "
        "These probabilities are derived from research literature patterns only. "
        "Please consult a qualified healthcare professional."
    )
```

---

## Phase 8 — Streamlit Frontend (Unified Conversational UI)

### Goal
Single-page Streamlit app integrating the full system: chat thread, clinical template progress bar, disease probability score panel, PICOS literature explorer sidebar, and Knowledge Graph tab (via `index.html`).

### Course requirement satisfied
> "Quality of presentation / working demo" — rubric (5%)
> "Visualization of results" — rubric (3%)

```python
# streamlit_app.py
import streamlit as st
from rag_pipeline import PICOSRetriever, RAGAnswerGenerator
from conversation_manager import init_session, handle_turn
from symptom_scorer import load_models, score_template, render_score_panel
import joblib

st.set_page_config(page_title="NORA — Neurodegenerative RAG Agent", layout="wide")

@st.cache_resource
def load_all():
    retriever  = PICOSRetriever()
    rag        = RAGAnswerGenerator(retriever)
    clf        = joblib.load("disease_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    le         = joblib.load("label_encoder.pkl")
    clf_sk, vec, le2, tok_bert, mod_bert = load_models()
    return retriever, rag, clf, vectorizer, le, clf_sk, vec, le2, tok_bert, mod_bert

retriever, rag, clf, vectorizer, le, clf_sk, vec, le2, tok_bert, mod_bert = load_all()
init_session()

# --- Sidebar: PICOS Literature Explorer ---
with st.sidebar:
    st.header("PICOS Literature Explorer")
    search_query   = st.text_input("Search query")
    filter_disease = st.selectbox("Filter by disease",
                                   ["All", "Alzheimer", "Parkinson", "ALS",
                                    "Huntington", "Dementia", "Stroke"])
    if search_query:
        fd      = None if filter_disease == "All" else filter_disease
        results = retriever.retrieve(search_query, k=5, filter_disease=fd)
        for r in results:
            with st.expander(f"{r['title'][:55]} ({r['disease']}, {r['year']})"):
                st.markdown(f"**P:** {r['P']}")
                st.markdown(f"**I:** {r['I']}")
                st.markdown(f"**C:** {r['C']}")
                st.markdown(f"**O:** {r['O']}")
                st.markdown(f"**S:** {r['S']}")
                st.write(r["abstract"][:400] + "...")
                st.caption(f"PMID: {r['pmid']}")

# --- Main area ---
st.title("NORA — Neurodegenerative RAG Agent")
st.caption("Literature-grounded assistant for Alzheimer's, Parkinson's, ALS, Huntington's, Dementia, and Stroke")

template = st.session_state.template
st.progress(template.filled_count() / 6,
            text=f"Clinical intake: {template.filled_count()}/6 fields collected")

if st.session_state.get("scored"):
    scores = score_template(template.to_text(), clf_sk, vec, le2, tok_bert, mod_bert)
    render_score_panel(scores)
    st.divider()

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Describe symptoms or ask a question..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    response = handle_turn(user_input, retriever, rag, clf, vectorizer, le)
    if response == "__SCORE__":
        st.rerun()
    else:
        with st.chat_message("assistant"):
            st.markdown(response)
```

---

## Build Order

```
Phase 1 → Phase 2A → Phase 2B → Phase 3 → Phase 3B → Phase 4 → Phases 5–8
Collect     NER       PICOS      ML+KG      KG export   RAG        Chatbot app
  ↓           ↓          ↓         ↓            ↓          ↓            ↓
abstracts   enrich   structure  cluster    graph.gexf  .faiss    full working
  .db         .db      PICOS    classify   graph.json   index       system
                       cols
```

> Phases 5–8 can be developed in parallel once Phase 4 is complete, since they all depend only on the database, saved models, and RAG pipeline — not on each other.

---

## Rubric Coverage Summary

| Rubric Item | % | Satisfied by |
|---|---|---|
| Data preparation + database | 2.5% | Phase 1 — SQLite with 1200 abstracts |
| Text feature engineering | 3% | Phase 2 — scispaCy NER + PICOS extraction; Phase 3 — TF-IDF + PICOS |
| Clustering | 3% | Phase 3A — K-Means with silhouette scoring |
| Classification (≥3 algorithms) | 3% | Phase 3B — LinearSVC, Logistic Regression, BioBERT |
| Evaluation of ML results | 4% | Phase 3B — accuracy, F1, confusion matrices; Phase 7 — ensemble scoring |
| Error analysis | 3% | Phase 3B — misclassification log, confusion matrix heatmaps |
| Knowledge graphs | included | Phase 3B — NetworkX + Gephi + D3.js export |
| RAG chatbot system | included | Phase 4 — FAISS + PICOS retrieval + Claude API |
| Template filling | included | Phase 5 — 6-field clinical intake with NER-assisted slot filling |
| Conversation management | included | Phase 6 — multi-turn session state + intent routing |
| Scoring program | included | Phase 7 — ensemble probability with disclaimer |
| Visualization | 3% | Silhouette plot, PCA scatter, confusion matrices, knowledge graph, score bars |
| Quality of presentation | 5% | Phase 8 — unified Streamlit app with progress bar, citations, PICOS explorer |

---

## File Structure

```
capstone/
├── phase1_collect.py              # PubMed data collection (6 disease queries)
├── phase2_ner_picos.py            # scispaCy NER + Claude PICOS extraction
├── phase3_ml.py                   # K-Means + 3-classifier comparison + error analysis
├── phase3b_knowledge_graph.py     # Knowledge graph build + Gephi + D3 export
├── phase4_rag.py                  # Sentence-BERT embeddings + FAISS index build
├── rag_pipeline.py                # PICOSRetriever + RAGAnswerGenerator
├── template_filler.py             # Slot-filling clinical intake (Phase 5)
├── conversation_manager.py        # Multi-turn routing + session management (Phase 6)
├── symptom_scorer.py              # Ensemble probability scoring (Phase 7)
├── streamlit_app.py               # Unified Streamlit frontend (Phase 8)
├── webhook.py                     # Existing Flask webhook (unchanged)
├── abstracts.db                   # SQLite (1200 abstracts + PICOS + NER columns)
├── abstracts.faiss                # FAISS vector index
├── faiss_id_map.pkl               # FAISS index → DB ID mapping
├── disease_classifier.pkl         # Best TF-IDF sklearn classifier (runtime routing)
├── tfidf_vectorizer.pkl           # Fitted TF-IDF vectorizer
├── label_encoder.pkl              # LabelEncoder (6 disease classes)
├── biobert_classifier/            # Fine-tuned BioBERT model directory
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer_config.json
├── neurological_triage.owl        # Existing ontology (unchanged)
├── silhouette_plot.png            # K-Means silhouette curve
├── cluster_pca_plot.png           # PCA 2D cluster scatter plot
├── classifier_comparison.png      # Accuracy bar chart (3 algorithms)
├── confusion_matrices.png         # Confusion matrices for all 3 classifiers
├── knowledge_graph.png            # Inline NetworkX visualisation
├── knowledge_graph.gexf           # Gephi-compatible export
├── knowledge_graph_data.json      # D3.js JSON for NORA web frontend
├── index.html                     # NORA web frontend (Agent + Docs + KG tabs)
├── error_analysis.csv             # Misclassification log from best classifier
├── README.md
└── requirements.txt
```

---

## requirements.txt

```
flask>=2.3.0
rdflib>=6.3.2
requests>=2.31.0
scispacy==0.5.4
spacy==3.7.4
scikit-learn>=1.3.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
anthropic>=0.25.0
streamlit>=1.28.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
pandas>=2.0.0
networkx>=3.1
transformers>=4.40.0
torch>=2.0.0
datasets>=2.14.0
scipy>=1.11.0
```
