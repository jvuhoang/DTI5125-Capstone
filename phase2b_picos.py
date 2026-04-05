"""
Phase 2B — PICOS Extraction using HuggingFace Zero-Shot Classification
=======================================================================
What this does:
  - Reads every abstract that has not yet had PICOS extracted
  - Splits each abstract into sentences using nltk
  - Uses a zero-shot classification pipeline with
    facebook/bart-large-mnli to label each sentence as one of
    the five PICOS elements:
      P — Population    : who was studied
      I — Intervention  : treatment, drug, or exposure studied
      C — Comparison    : what the intervention was compared to
      O — Outcome       : what was measured or observed
      S — Study design  : type of study
  - Processes sentences in batches for efficient GPU utilisation
  - For each PICOS element, picks the sentence with the highest
    confidence score as the extracted value
  - Writes all five fields back into the picos_* columns in the DB
  - Skips rows that already have picos_population filled
    so the script is safe to re-run if interrupted

Libraries used:
  - transformers  : HuggingFace pipeline for zero-shot classification
  - nltk          : sentence tokenisation
  - sqlite3       : read/write abstracts.db
  - torch         : GPU backend

Install:
  pip install transformers nltk torch

Run with:
  python phase2b_picos.py
"""

import sqlite3
import nltk
import torch
from transformers import pipeline

# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH         = "abstracts.db"
MODEL_NAME      = "facebook/bart-large-mnli"
SENTENCE_BATCH  = 32     # number of sentences sent to GPU at once
                         # increase to 64 if you have >8GB GPU VRAM
                         # decrease to 16 if you get CUDA out-of-memory errors
DB_COMMIT_EVERY = 50     # commit to database every N abstracts
LOG_EVERY       = 50     # print progress every N abstracts
MIN_CONFIDENCE  = 0.25   # minimum score to accept a sentence as a PICOS value

# ── PICOS candidate labels ────────────────────────────────────────────────────
#
# Natural language descriptions scored against each sentence.
# More descriptive labels produce better classification than
# single words like "population" or "outcome".

PICOS_LABELS = {
    "population":   "patients or participants who were studied, "
                    "including their disease, age group, or sample size",
    "intervention": "treatment, drug, therapy, surgery, or clinical "
                    "intervention that was applied or tested",
    "comparison":   "control group, placebo, or alternative treatment "
                    "that the intervention was compared against",
    "outcome":      "result, measurement, or endpoint that was observed "
                    "or reported such as symptoms, scores, or survival",
    "study_design": "type of study such as randomised controlled trial, "
                    "cohort study, case-control, meta-analysis, or review",
}


# ── Step 1: Setup ─────────────────────────────────────────────────────────────

def setup_nltk():
    """Download punkt tokeniser if not already present."""
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def load_model():
    """
    Load facebook/bart-large-mnli as a zero-shot classification pipeline.
    Automatically uses GPU if available, CPU otherwise.
    Sets batch_size on the pipeline so HuggingFace handles
    batching internally — suppresses the sequential GPU warning.
    """
    device = 0 if torch.cuda.is_available() else -1
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {device_name}")

    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=device,
        batch_size=SENTENCE_BATCH,   # tells pipeline to batch internally
    )
    print("Model loaded successfully")
    return classifier


# ── Step 2: Extract PICOS from one abstract ───────────────────────────────────

def extract_picos(classifier, abstract_text):
    """
    Extract all five PICOS elements from one abstract.

    Algorithm:
      1. Split abstract into sentences with nltk
      2. For each PICOS element, score ALL sentences at once
         against the element's candidate label using batched inference
      3. Pick the sentence with the highest confidence score
      4. Fall back to "not reported" if best score < MIN_CONFIDENCE

    Batching happens at the sentence level within each abstract —
    all sentences are scored in one GPU call per PICOS element
    rather than one GPU call per sentence.
    """
    sentences = [s.strip() for s in nltk.sent_tokenize(abstract_text)
                 if len(s.strip()) > 20]

    if not sentences:
        return {k: "not reported" for k in PICOS_LABELS}

    result = {}

    for picos_key, candidate_label in PICOS_LABELS.items():
        try:
            # Score all sentences at once — pipeline handles batching
            outputs = classifier(
                sentences,
                candidate_labels=[candidate_label],
                multi_label=False,
            )

            # outputs is a list of dicts when input is a list
            best_sentence = "not reported"
            best_score    = 0.0

            for sentence, output in zip(sentences, outputs):
                score = output["scores"][0]
                if score > best_score:
                    best_score    = score
                    best_sentence = sentence

            result[picos_key] = (
                best_sentence if best_score >= MIN_CONFIDENCE
                else "not reported"
            )

        except Exception as e:
            print(f"  [inference error] {picos_key}: {e}")
            result[picos_key] = "not reported"

    return result


# ── Step 3: Run PICOS extraction on all abstracts ─────────────────────────────

def run_picos(db_path=DB_PATH):
    """
    Main entry point.

    Fetches all rows where picos_population IS NULL,
    extracts PICOS from each abstract using batched GPU inference,
    and writes all five fields back to the database.

    Commits every DB_COMMIT_EVERY rows to protect against interruption.
    Re-running the script safely skips already-processed rows.

    Estimated time with T4 GPU:
      ~15-20 minutes for 1100 abstracts (vs 60-90 min without batching)
    Estimated time on Apple Silicon (MPS) CPU:
      ~40-60 minutes for 454 abstracts
    """
    setup_nltk()
    classifier = load_model()

    conn = sqlite3.connect(db_path)
    c    = conn.cursor()

    rows = c.execute("""
        SELECT id, abstract
        FROM abstracts
        WHERE picos_population IS NULL
          AND abstract IS NOT NULL
          AND abstract != ''
    """).fetchall()

    total = len(rows)
    print(f"\nAbstracts to process: {total}")

    if total == 0:
        print("Nothing to do — all abstracts already have PICOS extracted.")
        conn.close()
        return

    processed  = 0
    successful = 0
    failed     = 0

    for row_id, abstract_text in rows:
        try:
            picos = extract_picos(classifier, abstract_text)
            c.execute("""
                UPDATE abstracts SET
                    picos_population   = ?,
                    picos_intervention = ?,
                    picos_comparison   = ?,
                    picos_outcome      = ?,
                    picos_study_design = ?
                WHERE id = ?
            """, (
                picos.get("population",   "not reported"),
                picos.get("intervention", "not reported"),
                picos.get("comparison",   "not reported"),
                picos.get("outcome",      "not reported"),
                picos.get("study_design", "not reported"),
                row_id,
            ))
            successful += 1

        except Exception as e:
            c.execute("""
                UPDATE abstracts SET
                    picos_population   = 'extraction failed',
                    picos_intervention = 'extraction failed',
                    picos_comparison   = 'extraction failed',
                    picos_outcome      = 'extraction failed',
                    picos_study_design = 'extraction failed'
                WHERE id = ?
            """, (row_id,))
            failed += 1
            print(f"  [error] row {row_id}: {e}")

        processed += 1

        if processed % DB_COMMIT_EVERY == 0:
            conn.commit()

        if processed % LOG_EVERY == 0 or processed == total:
            pct = processed / total * 100
            print(f"  Progress: {processed}/{total} ({pct:.0f}%)"
                  f" | success={successful}"
                  f" | failed={failed}")

    conn.commit()
    conn.close()

    print(f"\n{'=' * 60}")
    print(f"PICOS extraction complete.")
    print(f"  Total processed : {processed}")
    print(f"  Successful      : {successful}")
    print(f"  Failed          : {failed}")


# ── Step 4: Verification helper ───────────────────────────────────────────────

def verify_picos(db_path=DB_PATH):
    """
    Inspect PICOS results after running.
    Prints coverage stats and one sample PICOS record per disease group.
    """
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()

    print("\nPICOS coverage:")
    total     = c.execute("SELECT COUNT(*) FROM abstracts").fetchone()[0]
    extracted = c.execute("""
        SELECT COUNT(*) FROM abstracts
        WHERE picos_population IS NOT NULL
          AND picos_population != 'extraction failed'
    """).fetchone()[0]
    failed = c.execute("""
        SELECT COUNT(*) FROM abstracts
        WHERE picos_population = 'extraction failed'
    """).fetchone()[0]
    pending = c.execute("""
        SELECT COUNT(*) FROM abstracts
        WHERE picos_population IS NULL
    """).fetchone()[0]

    print(f"  Extracted : {extracted} / {total}")
    print(f"  Failed    : {failed}")
    print(f"  Pending   : {pending}")

    print("\nSample PICOS records (1 per disease group):")
    diseases = c.execute(
        "SELECT DISTINCT disease_label FROM abstracts"
    ).fetchall()

    for (label,) in diseases:
        row = c.execute("""
            SELECT title,
                   picos_population, picos_intervention,
                   picos_comparison, picos_outcome, picos_study_design
            FROM abstracts
            WHERE disease_label = ?
              AND picos_population IS NOT NULL
              AND picos_population NOT IN ('not reported', 'extraction failed')
            LIMIT 1
        """, (label,)).fetchone()

        if not row:
            continue

        title, pop, interv, comp, outcome, study = row
        print(f"\n  [{label}]")
        print(f"  Title : {title[:65]}...")
        print(f"  P     : {pop[:100]}")
        print(f"  I     : {interv[:100]}")
        print(f"  C     : {comp[:100]}")
        print(f"  O     : {outcome[:100]}")
        print(f"  S     : {study[:100]}")

    conn.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_picos()
    verify_picos()
