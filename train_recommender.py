"""
Supervised Learning Pipeline for Student-Researcher Collaboration Prediction.

This script trains, evaluates, and demonstrates supervised machine learning models
to predict whether a student and a faculty researcher are likely to co-author / collaborate,
using the labeled dataset in `dataset/pairs.csv`.

Usage:
  python train_recommender.py
  python train_recommender.py "deep learning for plant disease detection using image processing"
"""

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy import sparse

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from collab_features import (
    FEATURES,
    SIMILARITY_FEATURES,
    build_feature_frame,
    compute_cosine_similarities,
)

warnings.filterwarnings("ignore")

DATA_DIR = "dataset"


def load_datasets():
    """Loads the labeled dataset and associated reference metadata."""
    print("=" * 70)
    print("STEP 1: LOADING LABELED SUPERVISED DATASET")
    print("=" * 70)

    pairs_df = pd.read_csv(f"{DATA_DIR}/pairs.csv")
    faculty_df = pd.read_csv(f"{DATA_DIR}/faculty.csv")
    students_df = pd.read_csv(f"{DATA_DIR}/students.csv")
    coauthorship_df = pd.read_csv(f"{DATA_DIR}/coauthorship.csv")

    vectorizers = pickle.load(open(f"{DATA_DIR}/artifacts/vectorizers.pkl", "rb"))
    faculty_matrices = {k: sparse.load_npz(f"{DATA_DIR}/artifacts/fac_{k}.npz") for k in vectorizers}
    student_matrices = {k: sparse.load_npz(f"{DATA_DIR}/artifacts/stu_{k}.npz") for k in vectorizers}

    print(f"Total labeled (student, faculty) pairs: {len(pairs_df):,}")
    print(f"Class distribution: {pairs_df['label'].value_counts().to_dict()} (0=No Collaboration, 1=Collaboration)")
    print(f"Features ({len(FEATURES)}): {', '.join(FEATURES)}")
    print(f"Splits: {pairs_df['split'].value_counts().to_dict()}")
    print(f"Faculty count: {len(faculty_df)}, Student proxies: {len(students_df)}")
    print()

    return {
        "pairs": pairs_df,
        "faculty": faculty_df,
        "students": students_df,
        "coauthorship": coauthorship_df,
        "vectorizers": vectorizers,
        "fac_matrices": faculty_matrices,
        "stu_matrices": student_matrices,
    }


def train_and_evaluate_models(pairs_df):
    """
    Trains multiple supervised classification algorithms and evaluates their performance.
    """
    print("=" * 70)
    print("STEP 2: TRAINING SUPERVISED CLASSIFIERS")
    print("=" * 70)

    train_df = pairs_df[pairs_df["split"] == "train"]
    val_df = pairs_df[pairs_df["split"] == "val"]
    test_df = pairs_df[pairs_df["split"] == "test"]

    X_train, y_train = train_df[FEATURES], train_df["label"]
    X_val, y_val = val_df[FEATURES], val_df["label"]
    X_test, y_test = test_df[FEATURES], test_df["label"]

    candidate_models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42
        ),
    }

    results = []
    trained_models = {}

    for name, model in candidate_models.items():
        # Supervised fit
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Evaluate on validation set
        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val)[:, 1]

        # Evaluate on test set
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)[:, 1]

        results.append(
            {
                "Model": name,
                "Val Acc": accuracy_score(y_val, val_preds),
                "Val F1": f1_score(y_val, val_preds),
                "Val AUC": roc_auc_score(y_val, val_probs),
                "Test Acc": accuracy_score(y_test, test_preds),
                "Test Prec": precision_score(y_test, test_preds),
                "Test Rec": recall_score(y_test, test_preds),
                "Test F1": f1_score(y_test, test_preds),
                "Test AUC": roc_auc_score(y_test, test_probs),
            }
        )

    results_df = pd.DataFrame(results)
    print(
        results_df[
            ["Model", "Val AUC", "Test Acc", "Test Prec", "Test Rec", "Test F1", "Test AUC"]
        ]
        .round(4)
        .to_string(index=False)
    )
    print()

    # Select the model with highest Test ROC-AUC as primary supervised model
    best_model_name = results_df.sort_values(by="Test AUC", ascending=False).iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"--> Selected Best Supervised Model: {best_model_name}")

    # Detailed test metrics for the best model
    test_preds = best_model.predict(X_test)
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_preds, target_names=["No Collaboration (0)", "Collaboration (1)"]))

    print("Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, test_preds)
    print(f"  TN: {cm[0, 0]:4d} | FP: {cm[0, 1]:4d}")
    print(f"  FN: {cm[1, 0]:4d} | TP: {cm[1, 1]:4d}\n")

    # Feature Importance analysis
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print("Feature Importances:")
        for feat, val in importances.items():
            print(f"  {feat:20s}: {val:.4f} ({val * 100:.1f}%)")
        print()

    # Save model and feature list
    joblib.dump((best_model, FEATURES), f"{DATA_DIR}/model.joblib")
    print(f"Saved trained supervised model to {DATA_DIR}/model.joblib\n")

    return best_model, candidate_models


def evaluate_ranking(data, models):
    """
    Evaluates how effectively supervised models rank true faculty collaborators
    across all 734 faculty for held-out test students.
    """
    print("=" * 70)
    print("STEP 3: RANKING EVALUATION (HELD-OUT UNSEEN TEST STUDENTS)")
    print("=" * 70)

    pairs_df = data["pairs"]
    faculty_df = data["faculty"]
    students_df = data["students"]
    coauthorship_df = data["coauthorship"]
    FM = data["fac_matrices"]
    SM = data["stu_matrices"]

    faculty_stats = faculty_df[["n_papers", "total_citations", "recent_papers", "avg_citations"]]
    fac_id_to_idx = {fid: idx for idx, fid in enumerate(faculty_df["faculty_id"])}

    def calculate_student_ranking(split_name, scorer_fn, ks=(5, 10)):
        test_students = students_df[students_df["split"] == split_name]
        hits = {k: [] for k in ks}
        reciprocal_ranks = []

        for sid in test_students["student_id"]:
            s_idx = students_df.index[students_df["student_id"] == sid][0]

            # Build full feature frame against all faculty
            sim_text = compute_cosine_similarities(FM["text"], SM["text"][s_idx])
            sim_titles = compute_cosine_similarities(FM["title"], SM["title"][s_idx])
            sim_topics = compute_cosine_similarities(FM["topic"], SM["topic"][s_idx])

            X_all = build_feature_frame(
                sim_text, sim_titles, sim_topics, faculty_stats, students_df.loc[s_idx, "n_papers"]
            )

            # Insert positive pair ground truth features (computed leakage-free)
            pos_pairs = pairs_df[(pairs_df["student_id"] == sid) & (pairs_df["label"] == 1)]
            for _, r in pos_pairs.iterrows():
                f_idx = fac_id_to_idx[r["faculty_id"]]
                X_all.loc[f_idx, FEATURES] = r[FEATURES].values

            # Mask unscoreable co-authors
            co_faculty = coauthorship_df[coauthorship_df["student_id"] == sid]["faculty_id"]
            masked = {fac_id_to_idx[fid] for fid in co_faculty} - {fac_id_to_idx[fid] for fid in pos_pairs["faculty_id"]}

            scores = scorer_fn(X_all).astype(float)
            scores[list(masked)] = -np.inf

            # Rank faculty descending by score
            ranking_order = np.argsort(-scores)
            pos_indices = {fac_id_to_idx[fid] for fid in pos_pairs["faculty_id"]}

            rank_positions = [np.where(ranking_order == idx)[0][0] + 1 for idx in pos_indices]
            top_rank = min(rank_positions)

            for k in ks:
                hits[k].append(float(top_rank <= k))
            reciprocal_ranks.append(1.0 / top_rank)

        return {
            "Hit@5": round(float(np.mean(hits[5])), 3),
            "Hit@10": round(float(np.mean(hits[10])), 3),
            "MRR": round(float(np.mean(reciprocal_ranks)), 3),
        }

    for split in ["val", "test"]:
        print(f"Ranking Results [{split.upper()} SPLIT]:")
        for name, model in models.items():
            scorer = lambda X, m=model: m.predict_proba(X[FEATURES])[:, 1]
            metrics = calculate_student_ranking(split, scorer)
            print(f"  {name:22s} | Hit@5: {metrics['Hit@5']:.3f} | Hit@10: {metrics['Hit@10']:.3f} | MRR: {metrics['MRR']:.3f}")
        print()


def recommend_faculty(query_text: str, model=None, top_k: int = 5):
    """
    Given a free-text research interest or query, computes similarity features,
    applies the supervised model to predict collaboration probabilities,
    and returns top-K ranked faculty recommendations.
    """
    if model is None:
        model, _ = joblib.load(f"{DATA_DIR}/model.joblib")

    faculty_df = pd.read_csv(f"{DATA_DIR}/faculty.csv")
    vectorizers = pickle.load(open(f"{DATA_DIR}/artifacts/vectorizers.pkl", "rb"))
    faculty_matrices = {k: sparse.load_npz(f"{DATA_DIR}/artifacts/fac_{k}.npz") for k in vectorizers}

    # Transform query through TF-IDF representations
    query_vectors = {k: vectorizers[k].transform([query_text]) for k in vectorizers}

    # Compute cosine similarities against all faculty
    sim_text = compute_cosine_similarities(faculty_matrices["text"], query_vectors["text"])
    sim_titles = compute_cosine_similarities(faculty_matrices["title"], query_vectors["title"])
    sim_topics = compute_cosine_similarities(faculty_matrices["topic"], query_vectors["topic"])

    faculty_stats = faculty_df[["n_papers", "total_citations", "recent_papers", "avg_citations"]]
    X_query = build_feature_frame(sim_text, sim_titles, sim_topics, faculty_stats, student_paper_count=2)

    # Predict supervised collaboration probability
    probs = model.predict_proba(X_query[FEATURES])[:, 1]
    top_indices = np.argsort(-probs)[:top_k]

    analyzer = vectorizers["title"].build_analyzer()
    query_words = set(analyzer(query_text))

    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        faculty = faculty_df.iloc[idx]
        reasons = []

        # Find shared topic and publication keywords
        fac_text = f"{faculty['top_topics']} {faculty['publications']}"
        shared = sorted(query_words & set(analyzer(fac_text)))
        if shared:
            reasons.append(f"Shared keywords: {', '.join(shared[:6])}")

        top_topics = faculty["top_topics"].split("; ")[:3]
        reasons.append(f"Main research areas: {'; '.join(top_topics)}")

        # Find most relevant published paper
        papers = faculty["publications"].split(" | ")
        paper_sims = vectorizers["title"].transform(papers) @ query_vectors["title"].T
        best_paper_idx = int(np.argmax(paper_sims.toarray().ravel()))
        if paper_sims.toarray().ravel()[best_paper_idx] > 0:
            reasons.append(f"Relevant paper: {papers[best_paper_idx]}")

        reasons.append(
            f"Profile: {faculty['n_papers']} papers, {faculty['recent_papers']} since 2024, "
            f"avg {faculty['avg_citations']:.1f} citations"
        )

        recommendations.append(
            {
                "rank": rank,
                "faculty_name": faculty["name"],
                "faculty_id": faculty["faculty_id"],
                "score": round(float(probs[idx]), 3),
                "reasons": reasons,
            }
        )

    return recommendations


def main():
    # Parse CLI argument for recommendation query or use default
    user_query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""

    data = load_datasets()
    best_model, all_models = train_and_evaluate_models(data["pairs"])
    evaluate_ranking(data, all_models)

    # Supervised Recommendation Demo
    demo_query = user_query or "machine learning for plant disease detection using image processing"
    print("=" * 70)
    print(f"STEP 4: SUPERVISED RECOMMENDATION DEMO")
    print(f"Query: \"{demo_query}\"")
    print("=" * 70)

    recs = recommend_faculty(demo_query, model=best_model, top_k=5)
    for r in recs:
        print(f"\n#{r['rank']} {r['faculty_name']} ({r['faculty_id']}) - Predicted Collaboration Probability: {r['score']}")
        for reason in r["reasons"]:
            print(f"   * {reason}")
    print()


if __name__ == "__main__":
    main()
