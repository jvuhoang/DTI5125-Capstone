"""
run_pipeline.py — Offline Build Script (run ONCE before starting the chatbot)
==============================================================================
Runs Phases 1–4 in sequence to build all the files the chatbot needs.
After this script completes, start the chatbot with:

    streamlit run streamlit_app.py

What this script produces:
  abstracts.db            — SQLite database of PubMed abstracts with NER + PICOS
  disease_classifier.pkl  — trained LinearSVC classifier
  tfidf_vectorizer.pkl    — fitted TF-IDF vectorizer
  label_encoder.pkl       — disease label encoder
  biobert_classifier/     — fine-tuned BioBERT (skipped on CPU-only machines)
  models/sentence_bert/   — Sentence-BERT cached locally (no re-download)
  abstracts.faiss         — FAISS vector index for RAG retrieval
  faiss_id_map.pkl        — FAISS row → SQLite ID mapping
  knowledge_graph.*       — NetworkX graph exports (Phase 3b)

Usage:
    python run_pipeline.py                # full pipeline
    python run_pipeline.py --skip-biobert # skip BioBERT fine-tuning (CPU-safe)
    python run_pipeline.py --from phase3  # resume from a specific phase
    python run_pipeline.py --only phase4  # run only one phase

No API keys required — Phase 2B uses facebook/bart-large-mnli
via HuggingFace (free, runs locally on CPU or GPU).
"""

import subprocess
import sys
import os
import time
import argparse

# ── Phase definitions ─────────────────────────────────────────────────────────

PHASES = [
    {
        "name":   "phase1",
        "label":  "Phase 1 — PubMed Abstract Collection",
        "script": "phase1_collect.py",
        "args":   [],
        "outputs": ["abstracts.db"],
    },
    {
        "name":   "phase2a",
        "label":  "Phase 2A — scispaCy NER",
        "script": "phase2_ner_picos.py",
        "args":   ["--part", "a"],
        "outputs": [],   # updates abstracts.db in-place
    },
    {
        "name":   "phase2b",
        "label":  "Phase 2B — PICOS Extraction (HuggingFace bart-large-mnli)",
        "script": "phase2b_picos.py",
        "args":   [],
        "outputs": [],
    },
    {
        "name":   "phase3",
        "label":  "Phase 3 — Clustering + Classification (sklearn + BioBERT)",
        "script": "phase3_ml.py",
        "args":   [],   # --skip-biobert injected dynamically if needed
        "outputs": [
            "disease_classifier.pkl",
            "tfidf_vectorizer.pkl",
            "label_encoder.pkl",
        ],
    },
    {
        "name":   "phase3b",
        "label":  "Phase 3b — Knowledge Graph",
        "script": "phase3b_knowledge_graph.py",
        "args":   [],
        "outputs": ["knowledge_graph.gexf", "knowledge_graph_data.json"],
    },
    {
        "name":   "phase4",
        "label":  "Phase 4 — RAG Embeddings + FAISS Index",
        "script": "phase4_rag.py",
        "args":   [],
        "outputs": ["abstracts.faiss", "faiss_id_map.pkl", "models/sentence_bert"],
    },
]

PHASE_NAMES = [p["name"] for p in PHASES]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _header(text: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {text}")
    print(f"{bar}")


def _check_outputs(phase: dict) -> bool:
    """Return True if all expected output files/dirs already exist."""
    for path in phase["outputs"]:
        if not os.path.exists(path):
            return False
    return bool(phase["outputs"])


def run_phase(phase: dict, skip_biobert: bool = False) -> bool:
    """
    Run a single phase script as a subprocess.
    Returns True on success, False on failure.
    """
    cmd  = [sys.executable, phase["script"]] + phase["args"]

    # Inject --skip-biobert for Phase 3 if requested
    if skip_biobert and phase["name"] == "phase3":
        cmd.append("--skip-biobert")

    _header(phase["label"])
    print(f"Command: {' '.join(cmd)}\n")

    start = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n✓ {phase['label']} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n✗ {phase['label']} FAILED (exit code {result.returncode})")
        return False


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="NORA offline build pipeline (run once before starting chatbot)"
    )
    parser.add_argument(
        "--skip-biobert",
        action="store_true",
        help="Skip BioBERT fine-tuning in Phase 3 (recommended for CPU-only machines)",
    )
    parser.add_argument(
        "--from",
        dest="from_phase",
        choices=PHASE_NAMES,
        default=None,
        help="Resume pipeline from this phase (skips earlier phases)",
    )
    parser.add_argument(
        "--only",
        dest="only_phase",
        choices=PHASE_NAMES,
        default=None,
        help="Run only this phase",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip phases whose output files already exist",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  NORA — Offline Build Pipeline")
    print("=" * 60)
    print(f"  skip-biobert  : {args.skip_biobert}")
    print(f"  from-phase    : {args.from_phase or 'beginning'}")
    print(f"  only-phase    : {args.only_phase or 'all'}")
    print(f"  skip-existing : {args.skip_existing}")

    # Phase 2B now uses HuggingFace locally — no API key needed
    # (bart-large-mnli is downloaded once from HuggingFace Hub, ~1.6 GB)

    # Determine which phases to run
    if args.only_phase:
        phases_to_run = [p for p in PHASES if p["name"] == args.only_phase]
    elif args.from_phase:
        start_idx     = PHASE_NAMES.index(args.from_phase)
        phases_to_run = PHASES[start_idx:]
    else:
        phases_to_run = PHASES

    # Run phases
    pipeline_start = time.time()
    successes      = 0
    failures       = 0

    for phase in phases_to_run:
        if args.skip_existing and _check_outputs(phase):
            _header(phase["label"])
            print(f"⏭  Skipping — output files already exist: {phase['outputs']}")
            successes += 1
            continue

        ok = run_phase(phase, skip_biobert=args.skip_biobert)
        if ok:
            successes += 1
        else:
            failures += 1
            print("\n⛔  Pipeline halted due to failure above.")
            print("    Fix the error and re-run with --from", phase["name"])
            break

    # Summary
    total_elapsed = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print(f"  Build complete  ({total_elapsed/60:.1f} min)")
    print(f"  Phases passed: {successes}  |  Failed: {failures}")
    print("=" * 60)

    if failures == 0:
        print("\n✅ All phases complete. Start the chatbot with:")
        print("     streamlit run streamlit_app.py\n")
    else:
        print("\n❌ Fix the failing phase and re-run.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
