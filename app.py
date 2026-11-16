"""
NexusScholar: Web Application for Supervised Faculty Recommendation.

This Flask application serves an interactive modern UI to search and rank faculty
collaborators based on predicted collaboration probabilities from the supervised ML model.

Usage:
  python app.py
  (Then open http://127.0.0.1:5000 in your browser)
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify
from train_recommender import recommend_faculty, DATA_DIR, load_datasets

app = Flask(__name__)

# Ensure trained supervised model exists before serving
MODEL_PATH = f"{DATA_DIR}/model.joblib"
if not os.path.exists(MODEL_PATH):
    print("Pre-trained model not found. Training supervised model...")
    from train_recommender import main as train_main
    train_main()


@app.route("/")
def index():
    """Renders the main interactive NexusScholar dashboard."""
    return render_template("index.html")


@app.route("/api/recommend", methods=["GET", "POST"])
def api_recommend():
    """
    API endpoint for supervised faculty recommendation.
    Accepts JSON body: { "query": "...", "top_k": 5 }
    Or query params: /api/recommend?query=...&top_k=5
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        query = data.get("query", "").strip()
        top_k = int(data.get("top_k", 5))
    else:
        query = request.args.get("query", "").strip()
        top_k = int(request.args.get("top_k", 5))

    if not query:
        return jsonify({"success": False, "error": "Query parameter cannot be empty."}), 400

    # Clamp top_k within reasonable bounds
    top_k = max(1, min(top_k, 25))

    try:
        recommendations = recommend_faculty(query, top_k=top_k)
        return jsonify({
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": recommendations
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n" + "=" * 60)
    print(f" NexusScholar Web App is running at:")
    print(f" http://127.0.0.1:{port}")
    print(f"=" * 60 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)
