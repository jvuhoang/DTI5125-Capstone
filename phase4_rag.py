"""
Phase 4 — RAG Pipeline: Embeddings + FAISS Index
=================================================
Encodes every abstract (title + text + key PICOS fields) using
Sentence-BERT (all-MiniLM-L6-v2) and stores vectors in a FAISS flat L2 index.

Outputs:
  abstracts.faiss     — FAISS vector index
  faiss_id_map.pkl    — maps FAISS index position → SQLite row ID

Run:
    python phase4_rag.py

Install:
    pip install sentence-transformers faiss-cpu
"""

import os
import sqlite3
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

DB_PATH          = "abstracts.db"
FAISS_PATH       = "abstracts.faiss"
MAP_PATH         = "faiss_id_map.pkl"
MODEL_NAME       = "all-MiniLM-L6-v2"
LOCAL_MODEL_DIR  = "./models/sentence_bert"   # saved here so chatbot never re-downloads
BATCH_SIZE       = 32


def load_abstracts(db_path: str = DB_PATH):
    """Load abstracts with their PICOS fields from the database."""
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()
    rows = c.execute("""
        SELECT id, title, abstract,
               picos_population, picos_intervention, picos_outcome
        FROM abstracts
        WHERE abstract IS NOT NULL AND abstract != ''
    """).fetchall()
    conn.close()
    print(f"Loaded {len(rows)} abstracts for embedding.")
    return rows


def build_embed_strings(rows: list) -> tuple[list, list]:
    """
    Concatenate title + abstract + key PICOS fields into a single string
    per abstract. The PICOS context makes retrieval clinically aware —
    queries about interventions match abstracts with relevant I-fields.
    """
    db_ids        = []
    texts_to_embed = []

    for row in rows:
        row_id, title, abstract, pop, intervention, outcome = row
        db_ids.append(row_id)
        embed_str = (
            f"{title or ''}. "
            f"{abstract or ''} "
            f"Population: {pop or ''}. "
            f"Intervention: {intervention or ''}. "
            f"Outcome: {outcome or ''}."
        )
        texts_to_embed.append(embed_str)

    return db_ids, texts_to_embed


def encode_and_index(texts: list, db_ids: list,
                     faiss_path: str = FAISS_PATH,
                     map_path:   str = MAP_PATH) -> None:
    """Encode texts with Sentence-BERT and build a FAISS flat L2 index."""
    # Load from local cache if available, otherwise download once then save
    if os.path.isdir(LOCAL_MODEL_DIR):
        print(f"Loading Sentence-BERT from local cache: {LOCAL_MODEL_DIR}")
        model = SentenceTransformer(LOCAL_MODEL_DIR)
    else:
        print(f"Downloading Sentence-BERT model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        model.save(LOCAL_MODEL_DIR)
        print(f"Model saved to {LOCAL_MODEL_DIR} — will load from disk on future runs.")

    print(f"Encoding {len(texts)} abstracts (batch_size={BATCH_SIZE})...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype("float32")
    print(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS index
    dimension = embeddings.shape[1]
    index     = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS index: {index.ntotal} vectors, dim={dimension}")

    # Save index and ID map
    faiss.write_index(index, faiss_path)
    with open(map_path, "wb") as f:
        pickle.dump(db_ids, f)

    print(f"Saved: {faiss_path}")
    print(f"Saved: {map_path}")


def verify_index(faiss_path: str = FAISS_PATH,
                 map_path:   str = MAP_PATH) -> None:
    """Quick sanity check — search for a test query."""
    print("\nVerifying FAISS index...")
    model = SentenceTransformer(LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else MODEL_NAME)
    index = faiss.read_index(faiss_path)
    with open(map_path, "rb") as f:
        id_map = pickle.load(f)

    test_queries = [
        "treatment for Parkinson's tremor",
        "Alzheimer's disease memory loss intervention",
        "ALS progression riluzole",
    ]

    for q in test_queries:
        vec             = model.encode([q]).astype("float32")
        distances, idxs = index.search(vec, 3)
        db_ids_found    = [id_map[i] for i in idxs[0]]
        print(f"  Query: '{q}'")
        print(f"  Top-3 DB IDs: {db_ids_found}  distances: {distances[0].tolist()}")


if __name__ == "__main__":
    rows              = load_abstracts()
    db_ids, texts     = build_embed_strings(rows)
    encode_and_index(texts, db_ids)
    verify_index()
    print("\nPhase 4 complete.")
