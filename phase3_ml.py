"""
Phase 3 — Clustering and Classification (3 Algorithms)
=======================================================
Part A: K-Means clustering with silhouette optimisation + PCA visualisation
Part B: Three disease classifiers — LinearSVC, Logistic Regression, Random Forest (optional: BioBERT)
        Full comparison with accuracy charts, confusion matrices, error analysis

Install:
    pip install scikit-learn matplotlib seaborn pandas scipy
    pip install transformers torch datasets

Run:
    python phase3_ml.py                    # full pipeline
    python phase3_ml.py --skip-biobert     # skip BioBERT (faster, no GPU needed)
    python phase3_ml.py --part cluster     # clustering only
    python phase3_ml.py --part classify    # classification only
"""

import argparse
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, classification_report,
    confusion_matrix, accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DB_PATH = "abstracts.db"

# ── Data loading ──────────────────────────────────────────────────────────────

# Canonical label names and any variant that should map to them.
# Handles cases where phase1 was run with different disease_label strings
# (e.g. "Alzheimer" vs "Alzheimer's Disease") across different runs.
LABEL_NORMALISE = {
    "Alzheimer":                        "Alzheimer's Disease",
    "Parkinson":                        "Parkinson's Disease",
    "Dementia":                         "Dementia and Mild Cognitive Impairment",
    "Huntington":                       "ALS and Huntington's Disease",
    "ALS":                              "ALS and Huntington's Disease",
    "Amyotrophic Lateral Sclerosis":    "ALS and Huntington's Disease",
}


def normalise_label(label: str) -> str:
    """Map label variants to canonical names."""
    return LABEL_NORMALISE.get(label, label)


def load_data(db_path: str = DB_PATH):
    """Load abstracts + PICOS fields from the database."""
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()
    rows = c.execute("""
        SELECT id, abstract, disease_label,
               picos_population, picos_intervention, picos_outcome
        FROM abstracts
        WHERE abstract IS NOT NULL AND abstract != ''
          AND disease_label IS NOT NULL
    """).fetchall()
    conn.close()

    ids      = [r[0] for r in rows]
    texts    = [r[1] for r in rows]
    # Normalise labels so variant spellings merge into the 5 canonical classes
    labels   = [normalise_label(r[2]) for r in rows]

    # Combine abstract text with key PICOS fields for richer TF-IDF features
    picos_text = [
        f"{r[3] or ''} {r[4] or ''} {r[5] or ''}"
        for r in rows
    ]
    combined = [f"{t} {p}" for t, p in zip(texts, picos_text)]

    print(f"Loaded {len(rows)} abstracts.")
    print("Class distribution (after normalisation):", Counter(labels))
    return ids, texts, labels, combined


def build_tfidf(combined: list):
    """Fit TF-IDF on combined text+PICOS strings."""
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(combined)
    print(f"TF-IDF matrix: {X.shape}")
    return X, vectorizer


# ── Part A: K-Means Clustering ────────────────────────────────────────────────

def run_clustering(X, labels: list) -> None:
    """K-Means over a range of k values; saves silhouette plot and PCA scatter."""
    print("\n── K-Means Clustering ──")

    k_range           = range(4, 14)
    silhouette_scores = []

    for k in k_range:
        km    = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        score = silhouette_score(X, km.labels_, sample_size=500)
        silhouette_scores.append(score)
        print(f"  k={k}: silhouette={score:.4f}")

    # Silhouette plot
    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), silhouette_scores, marker="o", color="steelblue")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Optimal cluster count — PubMed abstracts (TF-IDF + PICOS)")
    plt.tight_layout()
    plt.savefig("silhouette_plot.png", dpi=150)
    plt.close()
    print("Saved: silhouette_plot.png")

    best_k = list(k_range)[silhouette_scores.index(max(silhouette_scores))]
    print(f"\nBest k = {best_k}")

    # Final K-Means with best k
    km_final      = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(X)

    # Per-cluster analysis
    vectorizer_tmp = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1,2))
    vectorizer_tmp.fit_transform([" ".join(labels)])   # dummy; real names from stored vectorizer
    # Use cluster centres directly for top terms — vectorizer from build_tfidf is needed
    # (caller should pass it; here we use the cluster_labels for disease analysis only)
    for cluster_id in range(best_k):
        cluster_diseases = [labels[i] for i, c in enumerate(cluster_labels) if c == cluster_id]
        dominant = Counter(cluster_diseases).most_common(1)[0]
        print(f"  Cluster {cluster_id}: dominant={dominant[0]} n={len(cluster_diseases)}")

    # PCA 2D scatter
    print("\nGenerating PCA cluster plot...")
    X_dense = X.toarray()
    pca     = PCA(n_components=2, random_state=42)
    X_2d    = pca.fit_transform(X_dense)

    colors = plt.cm.tab10(np.linspace(0, 1, best_k))
    plt.figure(figsize=(10, 7))
    for cluster_id in range(best_k):
        mask = cluster_labels == cluster_id
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=[colors[cluster_id]], label=f"Cluster {cluster_id}",
                    alpha=0.5, s=20)
    plt.legend(fontsize=8)
    plt.title(f"K-Means Clusters (k={best_k}) — PCA 2D projection")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("cluster_pca_plot.png", dpi=150)
    plt.close()
    print("Saved: cluster_pca_plot.png")


# ── Part B: Classification ────────────────────────────────────────────────────

def train_sklearn_classifiers(X, y, le):
    """
    Train LinearSVC, Logistic Regression, and Random Forest.
    Returns results dict plus train/test splits.

    Random Forest notes:
    - TF-IDF matrices are sparse; RF converts them to dense internally, which
      uses more memory on large feature sets. max_features="sqrt" (default for
      classification) limits each split to sqrt(3000) ≈ 55 features, keeping
      training manageable.
    - n_estimators=200 gives a good accuracy/speed trade-off.
    - n_jobs=-1 uses all available CPU cores.
    """
    print("\n── sklearn Classifiers ──")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classifiers = {
        "LinearSVC":          LinearSVC(random_state=42, max_iter=2000),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000, C=1.0),
        "RandomForest":       RandomForestClassifier(
                                  n_estimators=200,
                                  max_features="sqrt",
                                  random_state=42,
                                  n_jobs=-1,
                              ),
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        results[name] = {"clf": clf, "y_pred": y_pred, "accuracy": acc}
        print(f"  {name}  |  Accuracy: {acc:.4f}")
        # Use only labels present in the test set to avoid target_names mismatch
        present = sorted(set(y_test) | set(y_pred))
        print(classification_report(y_test, y_pred,
                                    labels=present,
                                    target_names=[le.classes_[i] for i in present]))

    return results, X_train, X_test, y_train, y_test


def train_biobert_classifier(combined: list, y, le, test_size: float = 0.2):
    """Fine-tune BioBERT for sequence classification. Returns results entry."""
    print("\n── BioBERT Classifier ──")
    try:
        import torch
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                  TrainingArguments, Trainer)
        from torch.utils.data import Dataset as TorchDataset
    except ImportError:
        print("[ERROR] transformers or torch not installed.")
        print("  Run: pip install transformers torch")
        return None, None

    MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
    NUM_LABELS  = len(le.classes_)

    print(f"  Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Same split as sklearn
    _, test_idx = train_test_split(
        range(len(combined)), test_size=test_size, random_state=42, stratify=y
    )
    test_idx_set = set(test_idx)
    train_idx    = [i for i in range(len(combined)) if i not in test_idx_set]

    class AbstractDataset(TorchDataset):
        def __init__(self, indices):
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            i   = self.indices[idx]
            enc = tokenizer(
                combined[i],
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids":      enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels":         torch.tensor(y[i], dtype=torch.long),
            }

    train_dataset = AbstractDataset(train_idx)
    test_dataset  = AbstractDataset(list(test_idx))

    print(f"  Fine-tuning BioBERT on {len(train_idx)} examples...")
    model = AutoModelForSequenceClassification.from_pretrained(
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
        metric_for_best_model="accuracy",
        logging_dir="./logs",
        logging_steps=50,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels_arr = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float(accuracy_score(labels_arr, preds))}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Evaluate
    preds_out  = trainer.predict(test_dataset)
    bert_pred  = np.argmax(preds_out.predictions, axis=-1)
    bert_acc   = accuracy_score(y[list(test_idx)], bert_pred)
    y_test_bert = y[list(test_idx)]

    print(f"\n  BioBERT Accuracy: {bert_acc:.4f}")
    print(classification_report(y_test_bert, bert_pred, target_names=le.classes_))

    # Save model
    model.save_pretrained("biobert_classifier")
    tokenizer.save_pretrained("biobert_classifier")
    print("  BioBERT saved to ./biobert_classifier/")

    return (
        {"clf": model, "y_pred": bert_pred, "accuracy": bert_acc},
        y_test_bert,
    )


def plot_comparison(results: dict) -> None:
    """Bar chart comparing test accuracy across all classifiers."""
    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    colors = ["steelblue", "seagreen", "darkorange", "darkorchid"][:len(names)]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(names, accs, color=colors, edgecolor="black")
    plt.ylim(0.5, 1.0)
    plt.ylabel("Test Accuracy")
    plt.title("Disease Classifier Comparison — 6-class\n(TF-IDF baselines vs BioBERT)")
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig("classifier_comparison.png", dpi=150)
    plt.close()
    print("Saved: classifier_comparison.png")


def plot_confusion_matrices(results: dict, y_test_map: dict, le) -> None:
    """One confusion matrix heatmap per classifier."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        y_true = y_test_map.get(name, res.get("y_test"))
        cm = confusion_matrix(y_true, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_title(f"{name}\nacc={res['accuracy']:.3f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.tick_params(axis='x', rotation=35)

    plt.suptitle("Confusion Matrices — 6-class Neurodegenerative Disease Classifier",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    plt.close()
    print("Saved: confusion_matrices.png")


def run_error_analysis(results: dict, y_test_map: dict, combined: list,
                       y, le) -> None:
    """Log misclassified examples from the best classifier."""
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_pred = results[best_name]["y_pred"]
    y_true    = y_test_map[best_name]

    # Recover test indices for sklearn models (same split)
    _, test_indices = train_test_split(
        range(len(combined)), test_size=0.2, random_state=42, stratify=y
    )

    # For BioBERT the test set may differ in index ordering — use stored y_true directly
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != best_pred[i]:
            text_idx = test_indices[i] if i < len(test_indices) else i
            errors.append({
                "text":      combined[text_idx][:120] if text_idx < len(combined) else "",
                "actual":    le.inverse_transform([y_true[i]])[0],
                "predicted": le.inverse_transform([best_pred[i]])[0],
            })

    print(f"\nError Analysis — Best model: {best_name}")
    print(f"Misclassifications: {len(errors)} / {len(y_true)} "
          f"({len(errors)/len(y_true):.2%})")

    df = pd.DataFrame(errors)
    if not df.empty:
        print("\nBreakdown by actual → predicted:")
        print(df.groupby(["actual", "predicted"]).size().reset_index(name="count")
                .to_string(index=False))
        print("\nSample misclassified abstracts:")
        for e in errors[:5]:
            print(f"  Actual: {e['actual']} → Predicted: {e['predicted']}")
            print(f"  Text: {e['text']}...")

    df.to_csv("error_analysis.csv", index=False)
    print("Saved: error_analysis.csv")


def save_best_sklearn(results: dict, vectorizer, le) -> None:
    """
    Save sklearn models for runtime use in the chatbot.

    Artifacts produced:
      disease_classifier.pkl        — the best-performing sklearn model
                                      (LinearSVC, LR, or RandomForest)
      random_forest_classifier.pkl  — the Random Forest model specifically,
                                      so symptom_scorer.py can include it as
                                      a dedicated ensemble member alongside
                                      the best model and BioBERT.
      tfidf_vectorizer.pkl          — shared TF-IDF vectorizer
      label_encoder.pkl             — shared label encoder
    """
    sklearn_names = [n for n in results if n != "BioBERT"]
    if not sklearn_names:
        print("[WARN] No sklearn classifiers in results — nothing saved.")
        return

    # Save the best overall sklearn model (may or may not be RF)
    best_name = max(sklearn_names, key=lambda n: results[n]["accuracy"])
    joblib.dump(results[best_name]["clf"], "disease_classifier.pkl")
    joblib.dump(vectorizer,                "tfidf_vectorizer.pkl")
    joblib.dump(le,                        "label_encoder.pkl")
    print(f"\nBest sklearn model ({best_name}, acc={results[best_name]['accuracy']:.4f}) saved:")
    print("  disease_classifier.pkl")
    print("  tfidf_vectorizer.pkl")
    print("  label_encoder.pkl")

    # Always save RF separately for the ensemble (even if it isn't the overall best)
    if "RandomForest" in results:
        joblib.dump(results["RandomForest"]["clf"], "random_forest_classifier.pkl")
        rf_acc = results["RandomForest"]["accuracy"]
        print(f"\nRandom Forest (acc={rf_acc:.4f}) saved separately:")
        print("  random_forest_classifier.pkl")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Clustering + Classification")
    parser.add_argument("--part", choices=["cluster", "classify", "both"],
                        default="both", help="Which part to run (default: both)")
    parser.add_argument("--skip-biobert", action="store_true",
                        help="Skip BioBERT fine-tuning (faster, CPU-friendly)")
    parser.add_argument("--db", default=DB_PATH, help="Path to abstracts.db")
    args = parser.parse_args()

    # Load and prepare data
    ids, texts, labels, combined = load_data(args.db)

    # ── Remove classes with fewer than 2 samples ──────────────────────────────
    # train_test_split with stratify requires >= 2 members per class.
    # Rare / stray labels (e.g. a single abstract with a typo disease name)
    # would crash the split, so we drop them here and warn the user.
    label_counts = Counter(labels)
    rare_labels  = {lbl for lbl, cnt in label_counts.items() if cnt < 2}
    if rare_labels:
        print(f"\n[WARN] Dropping {len(rare_labels)} class(es) with fewer than "
              f"2 samples: {rare_labels}")
        keep_mask = [lbl not in rare_labels for lbl in labels]
        ids      = [v for v, k in zip(ids,      keep_mask) if k]
        texts    = [v for v, k in zip(texts,    keep_mask) if k]
        labels   = [v for v, k in zip(labels,   keep_mask) if k]
        combined = [v for v, k in zip(combined, keep_mask) if k]
        print(f"  Remaining abstracts: {len(labels)}")
        print(f"  Remaining classes  : {sorted(set(labels))}")

    X, vectorizer = build_tfidf(combined)
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    print(f"Label classes: {list(le.classes_)}")

    results    = {}
    y_test_map = {}

    if args.part in ("cluster", "both"):
        run_clustering(X, labels)

    if args.part in ("classify", "both"):
        # sklearn classifiers
        sklearn_results, X_train, X_test, y_train, y_test = train_sklearn_classifiers(
            X, y, le
        )
        results.update(sklearn_results)
        for name in sklearn_results:
            y_test_map[name] = y_test

        # BioBERT
        if not args.skip_biobert:
            bert_result, y_test_bert = train_biobert_classifier(combined, y, le)
            if bert_result:
                results["BioBERT"]    = bert_result
                y_test_map["BioBERT"] = y_test_bert
        else:
            print("\n[SKIP] BioBERT skipped (--skip-biobert flag set)")

        # Plots and analysis
        plot_comparison(results)
        plot_confusion_matrices(results, y_test_map, le)
        run_error_analysis(results, y_test_map, combined, y, le)
        save_best_sklearn(results, vectorizer, le)

    print("\nPhase 3 complete.")
