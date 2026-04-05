"""
webhook.py — Flask Webhook for Dialogflow ES
=============================================
Handles Dialogflow ES fulfillment requests from the existing chatbot.
Extends the original webhook with two new layers:

  1. Ontology layer  (existing) — queries neurological_triage.owl via rdflib
  2. RAG fallback    (new)      — if ontology returns no confident answer,
                                  fall through to the PICOS-aware RAG pipeline
  3. Classifier      (new)      — disease classifier pre-filters RAG retrieval
  4. Symptom scorer  (new)      — handles score-request intents

Environment variables:
    ANTHROPIC_API_KEY   — required for RAG layer
    PORT                — optional (default 5000)

Run locally:
    python webhook.py

Deploy on Render:
    Set start command to: gunicorn webhook:app
    Add ANTHROPIC_API_KEY as an environment secret
"""

import os
import json
import logging
from flask import Flask, request, jsonify
from rdflib import Graph, Namespace, RDF, RDFS, OWL
import joblib

from rag_pipeline import PICOSRetriever, RAGAnswerGenerator

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── Ontology setup ────────────────────────────────────────────────────────────

OWL_PATH = "neurological_triage.owl"
NEURO_NS  = Namespace("http://www.semanticweb.org/ontologies/neurological_triage#")

ontology_graph = Graph()
if os.path.exists(OWL_PATH):
    ontology_graph.parse(OWL_PATH, format="xml")
    log.info(f"Ontology loaded: {len(ontology_graph)} triples from {OWL_PATH}")
else:
    log.warning(f"OWL file not found at {OWL_PATH} — ontology layer disabled.")


# ── ML models (loaded once at startup) ───────────────────────────────────────

try:
    clf        = joblib.load("disease_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    le         = joblib.load("label_encoder.pkl")
    log.info("Disease classifier loaded.")
except FileNotFoundError:
    clf = vectorizer = le = None
    log.warning("Classifier files not found — disease pre-filtering disabled.")

# ── RAG pipeline (loaded once at startup) ─────────────────────────────────────

try:
    retriever = PICOSRetriever()
    rag       = RAGAnswerGenerator(retriever)
    log.info("RAG pipeline loaded.")
except Exception as e:
    retriever = rag = None
    log.warning(f"RAG pipeline not available: {e}")


# ── Ontology query helpers ────────────────────────────────────────────────────

def query_ontology(intent_name: str, parameters: dict) -> str | None:
    """
    Route a Dialogflow intent to the appropriate SPARQL/rdflib ontology query.
    Returns a formatted string response, or None if the intent is not handled.

    Extend this function to cover your existing 17 intents.
    """
    if not ontology_graph:
        return None

    disease = (
        parameters.get("disease")
        or parameters.get("Disease")
        or parameters.get("disease_name", "")
    )

    # ── Symptoms query ────────────────────────────────────────────────────────
    if "symptoms" in intent_name.lower() or "symptom" in intent_name.lower():
        if not disease:
            return "Please specify which disease you'd like to know about."
        results = _get_symptoms(disease)
        if results:
            return f"Common symptoms of {disease} include: {', '.join(results)}."
        return None

    # ── Risk factors query ────────────────────────────────────────────────────
    if "risk" in intent_name.lower():
        if not disease:
            return "Please specify which disease you'd like to know about."
        results = _get_risk_factors(disease)
        if results:
            return f"Key risk factors for {disease} include: {', '.join(results)}."
        return None

    # ── Treatment query ───────────────────────────────────────────────────────
    if "treatment" in intent_name.lower() or "therapy" in intent_name.lower():
        if not disease:
            return "Please specify which disease you'd like to know about."
        results = _get_treatments(disease)
        if results:
            return f"Common treatments for {disease} include: {', '.join(results)}."
        return None

    # ── Disease description ────────────────────────────────────────────────────
    if "what is" in intent_name.lower() or "describe" in intent_name.lower():
        if not disease:
            return None
        desc = _get_description(disease)
        if desc:
            return desc
        return None

    return None   # intent not handled by ontology


def _sparql_label_query(disease: str, property_uri: str) -> list[str]:
    """Generic helper: find objects linked to a disease node via a property."""
    results = []
    disease_lower = disease.lower().replace("'s", "").replace(" ", "_")
    for s in ontology_graph.subjects(RDF.type, OWL.Class):
        label = ontology_graph.value(s, RDFS.label)
        if label and disease_lower in str(label).lower():
            for obj in ontology_graph.objects(s, NEURO_NS[property_uri]):
                obj_label = ontology_graph.value(obj, RDFS.label) or str(obj).split("#")[-1]
                results.append(str(obj_label).replace("_", " "))
    return results[:8]   # cap at 8 items


def _get_symptoms(disease: str) -> list[str]:
    return _sparql_label_query(disease, "hasSymptom")


def _get_risk_factors(disease: str) -> list[str]:
    return _sparql_label_query(disease, "hasRiskFactor")


def _get_treatments(disease: str) -> list[str]:
    return _sparql_label_query(disease, "hasTreatment")


def _get_description(disease: str) -> str | None:
    disease_lower = disease.lower().replace("'s", "").replace(" ", "_")
    for s in ontology_graph.subjects(RDF.type, OWL.Class):
        label = ontology_graph.value(s, RDFS.label)
        if label and disease_lower in str(label).lower():
            comment = ontology_graph.value(s, RDFS.comment)
            if comment:
                return str(comment)
    return None


# ── RAG fallback ──────────────────────────────────────────────────────────────

def handle_rag_fallback(query_text: str, disease_filter: str = None) -> str:
    """
    Call the RAG pipeline and format the response with PICOS citation footnote.
    Returns a plain-text string suitable for Dialogflow fulfillmentText.
    """
    if not rag:
        return (
            "I'm sorry — the literature retrieval system is currently unavailable. "
            "Please try again later or rephrase your question."
        )

    try:
        result  = rag.answer(query_text, k=5, filter_disease=disease_filter)
        answer  = result["answer"]
        sources = result.get("sources", [])

        picos_note = ""
        summary    = result.get("picos_summary", [])
        if summary:
            top = summary[0]
            picos_note = (
                f"\n\nTop study — "
                f"Population: {top['P']} | "
                f"Intervention: {top['I']} | "
                f"Outcome: {top['O']} | "
                f"Design: {top['S']}"
            )

        citation_line = ""
        if sources:
            citation_line = f"\n\nSources: {', '.join(sources[:3])}"

        return f"{answer}{picos_note}{citation_line}"
    except Exception as e:
        log.error(f"RAG fallback error: {e}")
        return "I encountered an error retrieving literature. Please try rephrasing your question."


def predict_disease(text: str) -> str | None:
    """Predict disease label from query text for RAG pre-filtering."""
    if not (clf and vectorizer and le):
        return None
    try:
        vec  = vectorizer.transform([text])
        pred = clf.predict(vec)[0]
        return le.inverse_transform([pred])[0]
    except Exception:
        return None


# ── Webhook endpoint ──────────────────────────────────────────────────────────

@app.route("/webhook", methods=["POST"])
def webhook():
    """Main Dialogflow ES fulfillment endpoint."""
    try:
        body         = request.get_json(silent=True) or {}
        query_result = body.get("queryResult", {})
        intent_info  = query_result.get("intent", {})
        intent_name  = intent_info.get("displayName", "")
        query_text   = query_result.get("queryText", "")
        parameters   = query_result.get("parameters", {})

        log.info(f"Intent: '{intent_name}' | Query: '{query_text[:80]}'")

        response_text = None

        # ── Step 1: Try ontology layer ────────────────────────────────────────
        ontology_response = query_ontology(intent_name, parameters)

        if ontology_response and "Please specify" not in ontology_response:
            response_text = ontology_response
            log.info("Answered by ontology.")

        # ── Step 2: RAG fallback ───────────────────────────────────────────────
        if not response_text:
            disease_filter = predict_disease(query_text)
            log.info(f"RAG fallback — predicted disease: {disease_filter}")
            response_text  = handle_rag_fallback(query_text, disease_filter)
            log.info("Answered by RAG.")

        return jsonify({
            "fulfillmentText": response_text,
            "fulfillmentMessages": [
                {"text": {"text": [response_text]}}
            ],
        })

    except Exception as e:
        log.error(f"Webhook error: {e}", exc_info=True)
        return jsonify({
            "fulfillmentText": (
                "I'm sorry, I encountered an unexpected error. "
                "Please try again."
            )
        }), 200   # Always return 200 to Dialogflow


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Render deployment monitoring."""
    return jsonify({
        "status":    "ok",
        "ontology":  len(ontology_graph) > 0,
        "rag":       rag is not None,
        "classifier":clf is not None,
    })


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "NORA webhook is running."})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    log.info(f"Starting webhook on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
