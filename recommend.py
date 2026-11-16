"""
Supervised Recommender: Recommend Faculty for Student Research Profiles.

This standalone utility uses the trained supervised machine learning model
to predict collaboration probabilities and recommend matching faculty researchers.

Usage:
  python recommend.py "machine learning for medical image segmentation"
  python recommend.py "natural language processing for sentiment analysis"
  python recommend.py
"""

import sys
import os
import joblib
from train_recommender import recommend_faculty, DATA_DIR


def main():
    # Check if trained supervised model exists; if not, train it first
    model_path = f"{DATA_DIR}/model.joblib"
    if not os.path.exists(model_path):
        print("Trained model not found. Running training first...")
        from train_recommender import main as train_main
        train_main()
        print("\n" + "=" * 70 + "\n")

    # Get query from command line argument or prompt
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
    else:
        try:
            user_input = input("Enter student research interests / query: ").strip()
            query = user_input if user_input else "deep learning for crop disease detection"
        except (EOFError, KeyboardInterrupt):
            query = "deep learning for crop disease detection"

    print(f"\nSearching faculty matches for query: \"{query}\"\n")
    recommendations = recommend_faculty(query, top_k=5)

    print("-" * 70)
    for rec in recommendations:
        print(f"Rank #{rec['rank']}: {rec['faculty_name']} ({rec['faculty_id']})")
        print(f"Predicted Collaboration Probability: {rec['score'] * 100:.1f}%")
        for reason in rec["reasons"]:
            print(f"   * {reason}")
        print("-" * 70)


if __name__ == "__main__":
    main()
