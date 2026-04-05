"""
streamlit_app.py — NORA: Neurodegenerative RAG Agent
=====================================================
Unified Streamlit frontend that integrates the full NORA pipeline:
  - Chat thread with multi-turn memory
  - Clinical intake template progress bar
  - Disease probability score panel (ensemble of 3 classifiers)
  - PICOS literature explorer sidebar
  - Mode toggle (intake / direct RAG)
  - "New conversation" reset button

Run:
    streamlit run streamlit_app.py

Requirements:
    All Phase 1–7 outputs must exist:
      abstracts.db, abstracts.faiss, faiss_id_map.pkl,
      disease_classifier.pkl, tfidf_vectorizer.pkl, label_encoder.pkl
    Optional: biobert_classifier/ directory for ensemble scoring
"""

import streamlit as st
import joblib
import os

from rag_pipeline import PICOSRetriever, RAGAnswerGenerator
from template_filler import ClinicalTemplate
from conversation_manager import init_session, reset_session, handle_turn, get_greeting
from symptom_scorer import load_all_models

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NORA — Neurodegenerative RAG Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a3a5c;
        margin-bottom: 0.2rem;
    }
    .sub-caption {
        color: #555;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div {
        background-color: #1a3a5c;
    }
    .stChatMessage {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Resource loading (cached) ─────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading NORA models...")
def load_resources():
    """Load all models once and cache them for the lifetime of the app."""
    errors = []

    # Check for required files
    required = [
        "abstracts.db", "abstracts.faiss", "faiss_id_map.pkl",
        "disease_classifier.pkl", "tfidf_vectorizer.pkl", "label_encoder.pkl",
    ]
    for f in required:
        if not os.path.exists(f):
            errors.append(f"Missing required file: {f}")

    if errors:
        return None, "\n".join(errors)

    try:
        retriever  = PICOSRetriever()
        rag        = RAGAnswerGenerator(retriever)
        clf        = joblib.load("disease_classifier.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        le         = joblib.load("label_encoder.pkl")
        models     = load_all_models()

        return {
            "retriever":  retriever,
            "rag":        rag,
            "clf":        clf,
            "vectorizer": vectorizer,
            "le":         le,
            "models":     models,
        }, None
    except Exception as e:
        return None, str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(resources: dict) -> None:
    """Render the sidebar with controls and PICOS literature explorer."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/brain.png", width=60)
        st.markdown("## NORA")
        st.caption("Neurodegenerative RAG Agent")
        st.divider()

        # PICOS literature explorer
        st.markdown("### 📚 PICOS Literature Explorer")
        st.caption("Search the PubMed corpus by PICOS element")

        search_query   = st.text_input("Search query", placeholder="e.g. levodopa tremor")
        filter_disease = st.selectbox(
            "Filter by disease",
            ["All", "Alzheimer", "Parkinson", "ALS", "Huntington", "Dementia", "Stroke"],
        )
        search_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

        if search_query and resources:
            fd      = None if filter_disease == "All" else filter_disease
            results = resources["retriever"].retrieve(
                search_query, k=search_k, filter_disease=fd
            )
            if results:
                st.caption(f"Found {len(results)} result(s)")
                for r in results:
                    with st.expander(f"**{r['title'][:55]}...**\n{r['disease']} · {r['year']}"):
                        st.markdown(f"**P (Population):**   {r['P']}")
                        st.markdown(f"**I (Intervention):** {r['I']}")
                        st.markdown(f"**C (Comparison):**   {r['C']}")
                        st.markdown(f"**O (Outcome):**      {r['O']}")
                        st.markdown(f"**S (Study design):** {r['S']}")
                        st.write(r["abstract"][:400] + "...")
                        st.caption(f"PMID: {r['pmid']}")
            else:
                st.info("No results found. Try a different query.")
        elif search_query and not resources:
            st.error("Models not loaded — cannot search.")

        st.divider()

        # Reset button
        if st.button("🔄 New conversation", use_container_width=True):
            reset_session()
            st.rerun()

        # About
        with st.expander("About NORA"):
            st.write(
                "NORA is a literature-grounded clinical assistant built for the "
                "DTI5125 Capstone Project (Group 2). It uses:\n"
                "- **scispaCy NER** for biomedical entity extraction\n"
                "- **PICOS framework** for structured abstract retrieval\n"
                "- **FAISS + Sentence-BERT** for semantic search\n"
                "- **BioBERT + LinearSVC** for disease classification\n"
                "- **Claude API** for grounded answer generation\n\n"
                "Diseases: Alzheimer's · Parkinson's · ALS · Huntington's · "
                "Dementia · Stroke"
            )


# ── Main area ─────────────────────────────────────────────────────────────────

def render_main(resources: dict) -> None:
    """Render the main chat interface."""

    # Header
    st.markdown('<div class="main-header">🧠 NORA — Neurodegenerative RAG Agent</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-caption">Literature-grounded clinical assistant for '
        'Alzheimer\'s · Parkinson\'s · ALS · Huntington\'s · Dementia · Stroke</div>',
        unsafe_allow_html=True,
    )

    template = st.session_state.template

    # Thin mode indicator — only shows during intake, disappears in RAG mode
    if st.session_state.get("mode") != "rag":
        filled      = template.filled_count()
        progress_pct = filled / 6
        label = (
            f"Intake progress: {filled}/6  ✅ Ready to score"
            if template.is_complete()
            else f"Intake progress: {filled}/6 fields"
        )
        st.progress(progress_pct, text=label)

    # Chat history — score panel is rendered inline here, after the message
    # that triggered it, so it appears naturally in the conversation flow
    for msg in st.session_state.history:
        with st.chat_message(msg["role"],
                             avatar="🧠" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])

    # Chat input
    if resources:
        placeholder = (
            "Describe symptoms, or ask a literature question..."
            if st.session_state.get("mode") != "rag"
            else "Ask a question about treatments, outcomes, or research..."
        )
        if user_input := st.chat_input(placeholder):
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant", avatar="🧠"):
                with st.spinner("Thinking..."):
                    response = handle_turn(
                        user_input,
                        resources["retriever"],
                        resources["rag"],
                        resources["clf"],
                        resources["vectorizer"],
                        resources["le"],
                    )

                if response == "__SCORE__":
                    # The intro message was already added to history by handle_turn;
                    # rerun so the score panel renders below the chat in the right place
                    st.rerun()
                else:
                    st.markdown(response)
    else:
        st.error(
            "NORA could not load required model files. "
            "Please run the pipeline phases in order:\n"
            "1. `python phase1_collect.py`\n"
            "2. `python phase2_ner_picos.py`\n"
            "3. `python phase3_ml.py`\n"
            "4. `python phase3b_knowledge_graph.py`\n"
            "5. `python phase4_rag.py`\n"
            "Then restart Streamlit."
        )


# ── App entry point ───────────────────────────────────────────────────────────

def main():
    # Load resources
    resources, error = load_resources()

    # Initialise session state
    init_session()

    # Add greeting on first load
    if not st.session_state.history:
        st.session_state.history.append({
            "role":    "assistant",
            "content": get_greeting(),
        })

    # Render layout
    render_sidebar(resources)
    render_main(resources)

    if error:
        st.sidebar.error(f"Load error:\n{error}")


if __name__ == "__main__":
    main()
