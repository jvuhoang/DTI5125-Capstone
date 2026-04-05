"""
Phase 2 — Named Entity Recognition + PICOS Extraction
======================================================
Part A: Run scispaCy (en_ner_bc5cdr_md) on every abstract.
        Extracts DISEASE and CHEMICAL entity spans.
        Stores JSON string in ner_entities column.

Part B: Call Claude API on every abstract.
        Extracts Population, Intervention, Comparison, Outcome, Study design.
        Stores each PICOS element in its own column.

Install:
    pip install scispacy==0.5.4 spacy==3.7.4 anthropic
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

Run:
    python phase2_ner_picos.py
    python phase2_ner_picos.py --part a      # NER only
    python phase2_ner_picos.py --part b      # PICOS only

Environment:
    ANTHROPIC_API_KEY must be set for Part B
"""

import argparse
import json
import sqlite3
import time
import os

DB_PATH = "abstracts.db"

# ── Part A — scispaCy NER ─────────────────────────────────────────────────────

def run_ner(db_path: str = DB_PATH) -> None:
    """Extract DISEASE and CHEMICAL entities from every abstract using scispaCy."""
    try:
        import spacy
        nlp = spacy.load("en_ner_bc5cdr_md")
    except OSError:
        print("[ERROR] Could not load en_ner_bc5cdr_md.")
        print("Install with:")
        print("  pip install scispacy==0.5.4 spacy==3.7.4")
        print("  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
              "releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz")
        return

    conn = sqlite3.connect(db_path)
    c    = conn.cursor()

    rows = c.execute(
        "SELECT id, abstract FROM abstracts WHERE ner_entities IS NULL"
    ).fetchall()
    print(f"Running NER on {len(rows)} abstracts...")

    for i, (row_id, abstract_text) in enumerate(rows):
        if not abstract_text:
            c.execute("UPDATE abstracts SET ner_entities=? WHERE id=?",
                      (json.dumps([]), row_id))
            continue

        doc = nlp(abstract_text)
        entities = [
            {
                "text":  ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end":   ent.end_char,
            }
            for ent in doc.ents
        ]
        c.execute("UPDATE abstracts SET ner_entities=? WHERE id=?",
                  (json.dumps(entities), row_id))

        if (i + 1) % 100 == 0:
            conn.commit()
            print(f"  NER: {i+1}/{len(rows)}")

    conn.commit()
    conn.close()

    # Report
    conn = sqlite3.connect(db_path)
    done = conn.execute(
        "SELECT COUNT(*) FROM abstracts WHERE ner_entities IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    print(f"NER complete — {done} abstracts annotated.")


# ── Part B — PICOS Extraction via Claude ──────────────────────────────────────

PICOS_PROMPT = """\
Extract the PICOS elements from this medical abstract.
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


def extract_picos(client, abstract_text: str) -> dict | None:
    """Call Claude API to extract PICOS elements. Returns dict or None on failure."""
    if not abstract_text or len(abstract_text) < 50:
        return None
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{
                "role":    "user",
                "content": PICOS_PROMPT.format(abstract=abstract_text[:1500]),
            }],
        )
        raw = message.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            parts = raw.split("```")
            raw   = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] PICOS extraction: {e}")
        return None


def run_picos(db_path: str = DB_PATH) -> None:
    """Extract PICOS fields for all abstracts that don't yet have them."""
    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        print("[ERROR] anthropic package not installed. Run: pip install anthropic")
        return

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[ERROR] ANTHROPIC_API_KEY environment variable is not set.")
        return

    conn = sqlite3.connect(db_path)
    c    = conn.cursor()

    rows = c.execute("""
        SELECT id, abstract FROM abstracts
        WHERE picos_population IS NULL
          AND abstract IS NOT NULL
          AND abstract != ''
    """).fetchall()
    print(f"Extracting PICOS from {len(rows)} abstracts...")
    print("Estimated time: ~{:.0f} minutes".format(len(rows) * 0.1 / 60))

    success = 0
    for i, (row_id, abstract_text) in enumerate(rows):
        picos = extract_picos(client, abstract_text)

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
                picos.get("population",   "not reported"),
                picos.get("intervention", "not reported"),
                picos.get("comparison",   "not reported"),
                picos.get("outcome",      "not reported"),
                picos.get("study_design", "not reported"),
                row_id,
            ))
            success += 1
        else:
            # Mark as attempted so the script doesn't retry failed rows
            c.execute("""
                UPDATE abstracts SET picos_population='extraction_failed'
                WHERE id=?
            """, (row_id,))

        if (i + 1) % 50 == 0:
            conn.commit()
            print(f"  PICOS: {i+1}/{len(rows)} — {success} successful")

        time.sleep(0.1)   # Stay within API rate limits

    conn.commit()
    conn.close()
    print(f"PICOS extraction complete — {success}/{len(rows)} successfully extracted.")


def inspect_results(db_path: str = DB_PATH, n: int = 3) -> None:
    """Print sample PICOS results for verification."""
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()

    rows = c.execute("""
        SELECT title, disease_label, picos_population, picos_intervention,
               picos_comparison, picos_outcome, picos_study_design
        FROM abstracts
        WHERE picos_population IS NOT NULL
          AND picos_population != 'extraction_failed'
        ORDER BY RANDOM()
        LIMIT ?
    """, (n,)).fetchall()
    conn.close()

    print(f"\n{'='*60}")
    print("Sample PICOS extractions:")
    for row in rows:
        title, disease, pop, interv, comp, outcome, study = row
        print(f"\n[{disease}] {title[:65]}")
        print(f"  P: {pop}")
        print(f"  I: {interv}")
        print(f"  C: {comp}")
        print(f"  O: {outcome}")
        print(f"  S: {study}")


def print_counts(db_path: str = DB_PATH) -> None:
    """Print NER and PICOS completion counts."""
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()
    total = c.execute("SELECT COUNT(*) FROM abstracts").fetchone()[0]
    ner   = c.execute("SELECT COUNT(*) FROM abstracts WHERE ner_entities IS NOT NULL").fetchone()[0]
    picos = c.execute("""
        SELECT COUNT(*) FROM abstracts
        WHERE picos_population IS NOT NULL
          AND picos_population != 'extraction_failed'
    """).fetchone()[0]
    conn.close()
    print(f"\nTotal abstracts : {total}")
    print(f"NER complete    : {ner}")
    print(f"PICOS complete  : {picos}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: NER + PICOS extraction")
    parser.add_argument(
        "--part", choices=["a", "b", "both"], default="both",
        help="Which part to run: a=NER, b=PICOS, both=run A then B (default: both)"
    )
    parser.add_argument("--db", default=DB_PATH, help="Path to abstracts.db")
    parser.add_argument("--inspect", action="store_true",
                        help="Print sample results after running")
    args = parser.parse_args()

    if args.part in ("a", "both"):
        print("\n── Part A: scispaCy NER ──")
        run_ner(args.db)

    if args.part in ("b", "both"):
        print("\n── Part B: PICOS extraction ──")
        run_picos(args.db)

    print_counts(args.db)

    if args.inspect:
        inspect_results(args.db)
