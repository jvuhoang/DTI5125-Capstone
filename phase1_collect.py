"""
Phase 1 — PubMed Abstract Collection + SQLite Database
=======================================================
What this does:
  - Queries the PubMed E-utilities API using MeSH Boolean queries
    for five neurodegenerative disease groups:
      1. ALS and Huntington's Disease
      2. Alzheimer's Disease
      3. Dementia and Mild Cognitive Impairment
      4. Parkinson's Disease
      5. Stroke
  - Fetches up to MAX_PER_DISEASE abstracts per group
  - Parses each abstract from XML into structured fields
  - Stores everything in a local SQLite database (abstracts.db)
  - Schema includes placeholder columns for NER and PICOS fields
    so later phases can UPDATE rows in place without schema changes

Libraries used:
  - requests      : HTTP calls to the PubMed E-utilities API
  - xml.etree     : parsing PubMed XML responses
  - sqlite3       : local relational database (built into Python)
  - time          : rate limiting (PubMed allows max 3 requests/second)

Run with:
  python phase1_collect.py
"""

import requests
import xml.etree.ElementTree as ET
import sqlite3
import time

# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH          = "abstracts.db"
MAX_PER_DISEASE  = 250       # 250 x 5 groups = ~1250 abstracts total
RATE_LIMIT_SLEEP = 0.35      # seconds between requests (keeps under 3 req/s)

# ── Diagnosis / prognosis-focused MeSH Boolean queries ────────────────────────
#
# Keys are the full proper medical names used as disease_label values in the
# database, as classifier class names in Phase 3, and in the chatbot UI.
# Keep these labels consistent across ALL phases.

QUERIES = {
    "ALS and Huntington's Disease": (
        '("Amyotrophic Lateral Sclerosis"[MeSH] OR "Huntington Disease"[MeSH])'
        ' AND ("diagnosis"[MeSH] OR "prognosis"[MeSH] OR "symptoms"[tiab]'
        ' OR "risk factors"[MeSH] OR "biomarkers"[MeSH]'
        ' OR "disease progression"[MeSH])'
    ),
    "Alzheimer's Disease": (
        '"Alzheimer Disease"[MeSH]'
        ' AND ("early diagnosis"[tiab] OR "biomarkers"[MeSH]'
        ' OR "cognitive decline"[tiab] OR "risk factors"[MeSH]'
        ' OR "amyloid"[tiab] OR "tau"[tiab]'
        ' OR "neuropsychological tests"[MeSH])'
    ),
    "Dementia and Mild Cognitive Impairment": (
        '("Dementia"[MeSH] OR "Mild Cognitive Impairment"[MeSH])'
        ' AND ("diagnosis"[MeSH] OR "prognosis"[MeSH] OR "symptoms"[tiab]'
        ' OR "conversion"[tiab] OR "risk factors"[MeSH]'
        ' OR "neuroimaging"[MeSH])'
    ),
    "Parkinson's Disease": (
        '"Parkinson Disease"[MeSH]'
        ' AND ("diagnosis"[MeSH] OR "prognosis"[MeSH]'
        ' OR "motor symptoms"[tiab] OR "non-motor symptoms"[tiab]'
        ' OR "biomarkers"[MeSH] OR "risk factors"[MeSH]'
        ' OR "dopamine"[tiab] OR "UPDRS"[tiab])'
    ),
    "Stroke": (
        '"Stroke"[MeSH]'
        ' AND ("diagnosis"[MeSH] OR "prognosis"[MeSH]'
        ' OR "risk factors"[MeSH] OR "symptoms"[tiab]'
        ' OR "neurological deficits"[tiab] OR "imaging"[MeSH]'
        ' OR "functional outcome"[tiab])'
    ),
}

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


# ── Step 1: Create the database schema ────────────────────────────────────────

def create_database(db_path):
    """
    Create abstracts.db with a single 'abstracts' table.

    Columns are grouped into four logical sections:
      - Core metadata   : pmid, title, abstract, year, journal, disease_label
      - Phase 2A output : ner_entities (JSON string of entity spans)
      - Phase 2B output : picos_* columns (one per PICOS element)
      - Phase 5 output  : embedding_id (FAISS row index, filled in Phase 5)

    All PICOS and NER columns start as NULL so later phases can
    UPDATE rows without needing to ALTER the schema.
    """
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()

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

    conn.commit()
    print(f"Database ready: {db_path}")
    return conn


# ── Step 2: Fetch PubMed article IDs for a query ──────────────────────────────

def fetch_pubmed_ids(query, max_results):
    """
    Call the PubMed esearch endpoint and return a list of PMIDs.
    These IDs are passed one-by-one to efetch for the full record.
    Returns an empty list on any network or parse error.
    """
    params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
    }
    try:
        r = requests.get(ESEARCH_URL, params=params, timeout=15)
        r.raise_for_status()
        ids = r.json()["esearchresult"]["idlist"]
        return ids
    except Exception as e:
        print(f"  [esearch error] {e}")
        return []


# ── Step 3: Fetch and parse a single abstract by PMID ─────────────────────────

def fetch_abstract(pmid):
    """
    Call the PubMed efetch endpoint and parse the XML for one article.

    AbstractText can appear as multiple labelled sections
    (Background, Methods, Results, Conclusions) — we join them all
    into one string with a space separator.

    Returns a dict with keys: pmid, title, abstract, year, journal.
    Returns empty strings for any field that is missing in the XML.
    """
    params = {
        "db":      "pubmed",
        "id":      pmid,
        "rettype": "abstract",
        "retmode": "xml",
    }
    try:
        r = requests.get(EFETCH_URL, params=params, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)

        title = root.findtext(".//ArticleTitle") or ""

        abstract_parts = [
            t.text for t in root.findall(".//AbstractText") if t.text
        ]
        abstract = " ".join(abstract_parts)

        year    = root.findtext(".//PubDate/Year") or ""
        journal = root.findtext(".//Journal/Title") or ""

        return {
            "pmid":     pmid,
            "title":    title.strip(),
            "abstract": abstract.strip(),
            "year":     year.strip(),
            "journal":  journal.strip(),
        }

    except Exception as e:
        print(f"  [efetch error] pmid={pmid}: {e}")
        return {
            "pmid": pmid, "title": "",
            "abstract": "", "year": "", "journal": ""
        }


# ── Step 4: Insert a record into the database ─────────────────────────────────

def insert_abstract(cursor, record):
    """
    Insert one abstract record using INSERT OR IGNORE.
    Duplicate PMIDs are silently skipped — safe to re-run the script.
    Returns True if the row was inserted, False if it was a duplicate.
    """
    try:
        cursor.execute("""
            INSERT OR IGNORE INTO abstracts
                (pmid, title, abstract, year, journal, disease_label)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            record["pmid"],
            record["title"],
            record["abstract"],
            record["year"],
            record["journal"],
            record["disease_label"],
        ))
        return cursor.rowcount > 0
    except Exception as e:
        print(f"  [insert error] pmid={record['pmid']}: {e}")
        return False


# ── Step 5: Main collection loop ──────────────────────────────────────────────

def collect_all(db_path=DB_PATH):
    """
    Main entry point. For each disease group:
      1. Search PubMed for up to MAX_PER_DISEASE article IDs
      2. Fetch each abstract one at a time (with rate limiting)
      3. Skip records with no abstract text
      4. Insert into SQLite

    Commits every 50 inserts to avoid data loss on interruption.
    If the script is interrupted and re-run, already-inserted PMIDs
    are skipped automatically via INSERT OR IGNORE.
    """
    conn = create_database(db_path)
    c    = conn.cursor()

    total_inserted = 0

    for disease_label, query in QUERIES.items():
        print(f"\n{'─' * 60}")
        print(f"Disease group : {disease_label}")
        print(f"Query         : {query[:80]}...")

        ids = fetch_pubmed_ids(query, MAX_PER_DISEASE)
        print(f"Found {len(ids)} article IDs")

        inserted       = 0
        skipped        = 0
        empty_abstract = 0

        for i, pmid in enumerate(ids):
            record = fetch_abstract(pmid)
            record["disease_label"] = disease_label

            if not record["abstract"]:
                empty_abstract += 1
                time.sleep(RATE_LIMIT_SLEEP)
                continue

            ok = insert_abstract(c, record)
            if ok:
                inserted += 1
            else:
                skipped += 1

            if (i + 1) % 50 == 0:
                conn.commit()
                print(f"  Progress : {i+1}/{len(ids)} fetched"
                      f" | inserted={inserted}"
                      f" | skipped={skipped}")

            time.sleep(RATE_LIMIT_SLEEP)

        conn.commit()
        total_inserted += inserted
        print(f"Done — inserted={inserted}"
              f" | skipped(duplicate)={skipped}"
              f" | skipped(no abstract)={empty_abstract}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Collection complete.")
    print(f"Total new records inserted this run: {total_inserted}")
    print(f"\nFull breakdown by disease group:")

    for label in QUERIES:
        count = c.execute(
            "SELECT COUNT(*) FROM abstracts WHERE disease_label=?", (label,)
        ).fetchone()[0]
        print(f"  {label:45s}: {count} abstracts")

    total = c.execute("SELECT COUNT(*) FROM abstracts").fetchone()[0]
    print(f"\n  {'TOTAL':45s}: {total} abstracts")
    print(f"\n  Database file: {db_path}")

    conn.close()


# ── Step 6: Verification helper ───────────────────────────────────────────────

def verify_database(db_path=DB_PATH):
    """
    Quick sanity check after collection.
    Prints the schema and two sample rows per disease group.
    Call this independently to inspect the database at any time:

      from phase1_collect import verify_database
      verify_database()
    """
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()

    print("\nSchema columns:")
    for row in c.execute("PRAGMA table_info(abstracts)"):
        nullable = "NULL" if not row[3] else "NOT NULL"
        print(f"  {row[1]:25s} {row[2]:10s} {nullable}")

    print("\nSample rows (first 2 per disease group):")
    for label in QUERIES:
        rows = c.execute("""
            SELECT pmid, year, journal, title
            FROM abstracts
            WHERE disease_label = ?
            LIMIT 2
        """, (label,)).fetchall()
        print(f"\n  [{label}]")
        for pmid, year, journal, title in rows:
            print(f"    PMID {pmid} ({year}) [{journal[:35]}]")
            print(f"    {title[:70]}...")

    conn.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    collect_all()
    verify_database()