"""
Phase 6 — Recommender Sub-System (Surprise Library)
====================================================
What this does:
  This is a sub-system of a recommender system as specified
  in the professor's instructions. Rather than forcing a
  collaborative filtering analogy onto clinical literature,
  this implements a PICOS profile similarity recommender:

  Given a user's symptom profile (filled by the template
  filler in Phase 7), find the most similar abstracts in
  the corpus by comparing PICOS profiles using matrix
  factorisation techniques from the Surprise library.

  Part A — Dataset Construction
    - Builds an interaction matrix from the database where:
        "User"   = disease group (5 groups)
        "Item"   = unique PICOS intervention string
        "Rating" = normalised frequency of that intervention
                   appearing in abstracts for that disease group
    - This is a content-based sub-system, not collaborative
      filtering — we are finding similar intervention profiles
      across disease groups

  Part B — Three Surprise Algorithms
    - SVD      : matrix factorisation (gold standard)
    - KNNBasic : k-nearest neighbours on item similarity
    - NMF      : non-negative matrix factorisation
    - Each evaluated with k-fold cross-validation
      (k is set automatically based on dataset size)
    - Results compared in a bar chart

  Part C — Similarity Recommender
    - Given a disease group, finds the top-N interventions
      most likely to be relevant based on learned profiles
    - Returns ranked list with supporting PMIDs

  Part D — Evaluation and Visualisation
    - RMSE and MAE comparison bar chart across 3 algorithms
    - Saves recommender_comparison.png
    - Saves the best model to recommender.pkl

Libraries used:
  - scikit-surprise : SVD, KNNBasic, NMF, cross_validate
  - sqlite3         : read abstracts and PICOS fields
  - pandas          : build interaction dataframe
  - matplotlib      : evaluation bar chart
  - joblib          : save best model

Install:
  pip install scikit-surprise pandas matplotlib joblib

Run with:
  python phase6_recommender.py
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from collections import defaultdict, Counter

from surprise import Dataset, Reader, SVD, KNNBasic, NMF
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split as surprise_split

# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH       = "abstracts.db"
MODEL_PATH    = "recommender.pkl"
CHART_PATH    = "recommender_comparison.png"
MIN_FREQ      = 1       # lowered to 1 to maximise dataset size
                        # increase to 2 or 3 if you have a large corpus
N_FOLDS       = 5       # cross-validation folds — auto-reduced if needed
TOP_N         = 5       # number of recommendations to return
RATING_SCALE  = (1, 5)  # Surprise requires an explicit rating scale


# ── Part A: Build the interaction dataset ─────────────────────────────────────

def build_dataset(db_path=DB_PATH):
    """
    Build an interaction matrix suitable for Surprise.

    Rows    = disease groups (5 groups as "users")
    Columns = PICOS intervention strings (as "items")
    Values  = normalised frequency score (1-5 scale)

    The score for (disease, intervention) is:
      score = 1 + 4 * (count / max_count_in_group)
    scaled to [1, 5] to match Surprise's rating scale.
    """
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()

    rows = c.execute("""
        SELECT pmid, disease_label, picos_intervention
        FROM abstracts
        WHERE picos_intervention IS NOT NULL
          AND picos_intervention NOT IN
              ('not reported', 'extraction failed', '')
          AND disease_label IS NOT NULL
    """).fetchall()
    conn.close()

    print(f"Loaded {len(rows)} abstracts with interventions")

    # Count (disease, intervention) co-occurrences
    counts   = defaultdict(int)
    item_map = defaultdict(list)

    for pmid, disease, intervention in rows:
        intervention = intervention.strip().lower()[:80]
        counts[(disease, intervention)] += 1
        item_map[intervention].append(pmid)

    # Filter by minimum frequency
    intervention_totals = Counter()
    for (disease, intervention), count in counts.items():
        intervention_totals[intervention] += count

    valid_interventions = {
        interv for interv, total in intervention_totals.items()
        if total >= MIN_FREQ
    }
    print(f"Unique interventions (freq >= {MIN_FREQ}): "
          f"{len(valid_interventions)}")

    # Compute max count per disease group for normalisation
    max_per_disease = defaultdict(int)
    for (disease, intervention), count in counts.items():
        if intervention in valid_interventions:
            max_per_disease[disease] = max(
                max_per_disease[disease], count
            )

    # Build interaction records scaled to [1, 5]
    records = []
    for (disease, intervention), count in counts.items():
        if intervention not in valid_interventions:
            continue
        max_count = max_per_disease[disease]
        if max_count == 0:
            continue
        score = 1 + 4 * (count / max_count)
        records.append({
            "user":   disease,
            "item":   intervention,
            "rating": round(score, 2),
        })

    df = pd.DataFrame(records)
    print(f"Interaction matrix: {len(df)} records")
    print(f"  Disease groups : {df['user'].nunique()}")
    print(f"  Interventions  : {df['item'].nunique()}")
    print(f"\nSample records:")
    print(df.head(8).to_string(index=False))

    reader = Reader(rating_scale=RATING_SCALE)
    data   = Dataset.load_from_df(df[["user", "item", "rating"]], reader)

    return df, data, item_map


# ── Part B: Train and evaluate three algorithms ───────────────────────────────

def train_algorithms(data, df):
    """
    Train SVD, KNNBasic, and NMF on the interaction dataset.

    The number of cross-validation folds is automatically reduced
    if the dataset is too small — Surprise requires n_splits to be
    less than the number of ratings. We use at most N_FOLDS but
    fall back to 2 if needed.

    Returns:
      results   : dict of {name: {rmse, mae, algo}}
      best_name : name of the best algorithm by RMSE
    """
    print(f"\n{'─' * 60}")
    print("Part B — Training 3 Surprise algorithms")

    # Safety check — reduce folds if dataset is too small
    n_ratings = len(df)
    n_folds   = min(N_FOLDS, n_ratings - 1)
    n_folds   = max(n_folds, 2)
    print(f"Dataset size: {n_ratings} ratings | Using {n_folds}-fold CV")

    algorithms = {
        "SVD":      SVD(n_factors=50, n_epochs=20, random_state=42),
        "KNNBasic": KNNBasic(
            k=min(10, n_ratings // n_folds),
            sim_options={"name": "cosine", "user_based": False}
        ),
        "NMF":      NMF(n_factors=15, n_epochs=50, random_state=42),
    }

    results = {}
    for name, algo in algorithms.items():
        print(f"\n  Training {name}...")
        try:
            cv_results = cross_validate(
                algo, data,
                measures=["RMSE", "MAE"],
                cv=n_folds,
                verbose=False,
            )
            mean_rmse = np.mean(cv_results["test_rmse"])
            mean_mae  = np.mean(cv_results["test_mae"])
            results[name] = {
                "rmse": mean_rmse,
                "mae":  mean_mae,
                "algo": algo,
            }
            print(f"    RMSE: {mean_rmse:.4f} | MAE: {mean_mae:.4f}")

        except Exception as e:
            print(f"    [SKIP] {name} failed: {e}")
            continue

    if not results:
        raise RuntimeError(
            "All algorithms failed. Check your dataset size."
        )

    best_name = min(results, key=lambda n: results[n]["rmse"])
    print(f"\n  Best algorithm: {best_name} "
          f"(RMSE={results[best_name]['rmse']:.4f})")

    return results, best_name


# ── Part C: Similarity recommender ────────────────────────────────────────────

def build_similarity_recommender(df, data, best_algo_name, results):
    """
    Train the best algorithm on the full dataset.
    Returns the trained algorithm and full trainset.
    """
    print(f"\n{'─' * 60}")
    print("Part C — Building similarity recommender")

    trainset = data.build_full_trainset()
    algo     = results[best_algo_name]["algo"]
    algo.fit(trainset)

    print(f"  Trained {best_algo_name} on full dataset")
    print(f"  Trainset size: {trainset.n_ratings} interactions")

    return algo, trainset


def recommend(algo, df, disease_label, item_map, top_n=TOP_N):
    """
    Generate top-N intervention recommendations for a disease group.

    Predicts scores for interventions not yet seen for this disease
    and returns the top-N highest scoring ones with supporting PMIDs.
    """
    seen = set(df[df["user"] == disease_label]["item"].tolist())
    all_items = df["item"].unique()

    predictions = []
    for item in all_items:
        if item in seen:
            continue
        try:
            pred = algo.predict(disease_label, item)
            predictions.append((item, pred.est))
        except Exception:
            continue

    predictions.sort(key=lambda x: x[1], reverse=True)

    recommendations = []
    for intervention, score in predictions[:top_n]:
        recommendations.append({
            "intervention":    intervention,
            "predicted_score": round(score, 3),
            "pmids":           item_map.get(intervention, [])[:3],
        })

    return recommendations


# ── Part D: Evaluation visualisation ──────────────────────────────────────────

def visualise_results(results, algo, df, item_map,
                      chart_path=CHART_PATH):
    """
    Bar chart comparing RMSE and MAE across all 3 algorithms.
    Sample recommendations for each disease group.
    """
    print(f"\n{'─' * 60}")
    print("Part D — Evaluation and Visualisation")

    names = list(results.keys())
    rmses = [results[n]["rmse"] for n in names]
    maes  = [results[n]["mae"]  for n in names]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, rmses, width,
                   label="RMSE", color="#1D9E75", edgecolor="white")
    bars2 = ax.bar(x + width/2, maes,  width,
                   label="MAE",  color="#378ADD", edgecolor="white")

    ax.set_ylabel("Error")
    ax.set_title(
        "Recommender sub-system evaluation — "
        "SVD vs KNNBasic vs NMF"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()

    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=10
        )

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.show()
    print(f"Saved: {chart_path}")

    # Sample recommendations per disease group
    print(f"\nSample recommendations (top {TOP_N} per disease group):")
    for disease in df["user"].unique():
        recs = recommend(algo, df, disease, item_map, top_n=TOP_N)
        print(f"\n  [{disease}]")
        if not recs:
            print("    No recommendations generated")
            continue
        for i, rec in enumerate(recs, 1):
            print(f"    {i}. {rec['intervention'][:60]}"
                  f" (score={rec['predicted_score']})"
                  f" — {len(rec['pmids'])} supporting abstracts")


# ── Save best model ───────────────────────────────────────────────────────────

def save_model(algo, df, item_map, model_path=MODEL_PATH):
    """
    Save the trained recommender, interaction dataframe,
    and item map for use in Phase 8 (conversation manager).
    """
    joblib.dump({
        "algo":     algo,
        "df":       df,
        "item_map": item_map,
    }, model_path)
    print(f"\nRecommender saved to: {model_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Part A — build dataset
    df, data, item_map = build_dataset()

    # Part B — train and evaluate 3 algorithms
    results, best_name = train_algorithms(data, df)

    # Part C — build recommender on full data
    algo, trainset = build_similarity_recommender(
        df, data, best_name, results
    )

    # Part D — visualise and sample recommendations
    visualise_results(results, algo, df, item_map)

    # Save model
    save_model(algo, df, item_map)

    print(f"\n{'=' * 60}")
    print("Phase 6 complete.")
    print(f"  recommender_comparison.png — evaluation chart")
    print(f"  recommender.pkl            — saved model")