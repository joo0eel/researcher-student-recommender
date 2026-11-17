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
from train_recommender import recommend_faculty, get_recommender_assets, DATA_DIR, load_datasets

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


@app.route("/api/faculty", methods=["GET"])
def api_faculty():
    """
    Search and explore faculty members, their OpenAlex links, and past works.
    Query params: search, page, per_page
    """
    search = request.args.get("search", "").strip().lower()
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 12))

    faculty_df, _, _, _ = get_recommender_assets()
    filtered = faculty_df
    if search:
        mask = (
            filtered["name"].astype(str).str.lower().str.contains(search, na=False) |
            filtered["top_topics"].astype(str).str.lower().str.contains(search, na=False) |
            filtered["publications"].astype(str).str.lower().str.contains(search, na=False)
        )
        filtered = filtered[mask]

    total = len(filtered)
    start = max(0, (page - 1) * per_page)
    end = start + per_page
    rows = filtered.iloc[start:end]

    items = []
    for _, f in rows.iterrows():
        papers = [p.strip() for p in str(f.get("publications", "")).split(" | ") if p.strip()]
        openalex_url = str(f.get("openalex_id", "")).strip()
        if openalex_url and not openalex_url.startswith("http"):
            openalex_url = f"https://openalex.org/{openalex_url}"

        items.append({
            "faculty_id": f["faculty_id"],
            "name": f["name"],
            "openalex_url": openalex_url,
            "n_papers": int(f["n_papers"]),
            "total_citations": int(f["total_citations"]),
            "recent_papers": int(f["recent_papers"]),
            "avg_citations": round(float(f["avg_citations"]), 1),
            "top_topics": [t.strip() for t in str(f.get("top_topics", "")).split(";") if t.strip()][:5],
            "past_publications": papers[:6]
        })

    return jsonify({
        "success": True,
        "total": total,
        "page": page,
        "per_page": per_page,
        "items": items
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n" + "=" * 60)
    print(f" NexusScholar Web App is running at:")
    print(f" http://127.0.0.1:{port}")
    print(f"=" * 60 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)
