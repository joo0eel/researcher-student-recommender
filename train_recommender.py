"""
Train + evaluate the recommender on dataset/pairs.csv, then recommend for any free-text profile.

  python train_recommender.py                       # train, evaluate, demo
  python train_recommender.py "deep learning for crop disease detection"
"""
import sys, pickle, re
import numpy as np, pandas as pd, joblib
from scipy import sparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from collab_features import FEATURES, frame, one_vs_all, NonNegativeScorer

D = "dataset"
pairs = pd.read_csv(f"{D}/pairs.csv"); fac = pd.read_csv(f"{D}/faculty.csv")
stu = pd.read_csv(f"{D}/students.csv"); co = pd.read_csv(f"{D}/coauthorship.csv")
V = pickle.load(open(f"{D}/artifacts/vectorizers.pkl", "rb"))
FM = {k: sparse.load_npz(f"{D}/artifacts/fac_{k}.npz") for k in V}
SM = {k: sparse.load_npz(f"{D}/artifacts/stu_{k}.npz") for k in V}
FSTAT = fac[["n_papers", "total_citations", "recent_papers", "avg_citations"]]
fidx = {f: i for i, f in enumerate(fac.faculty_id)}

# ---------------------------------------------------------------- train
SIMS = ["sim_text", "sim_titles", "sim_topics"]
tr, va, te = (pairs[pairs.split == s] for s in ("train", "val", "test"))
models = {   # name -> (model, feature columns)
    "non-negative linear (similarity features)": (NonNegativeScorer(), SIMS),
    "logistic_regression (similarity features)": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), SIMS),
    "gradient_boosting (all features)": (GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                                      subsample=0.8, random_state=0), FEATURES),
}
for name, (m, cols) in models.items():
    m.fit(tr[cols], tr.label)
    print(f"{name:44s} AUC val={roc_auc_score(va.label, m.predict_proba(va[cols])[:,1]):.3f} "
          f"test={roc_auc_score(te.label, m.predict_proba(te[cols])[:,1]):.3f}")
# Final model: non-negative linear scorer on similarity features. It beats the text-similarity baseline on
# validation AND test ranking, and unlike plain logistic regression its weights stay interpretable (all >= 0).
model, MCOLS = models["non-negative linear (similarity features)"]
joblib.dump((model, MCOLS), f"{D}/model.joblib")
coef = dict(zip(MCOLS, model.coef_.round(3)))
print("\nFinal model coefficients (standardised):", coef)


# ---------------------------------------------- ranking evaluation (all faculty)
def rank_metrics(split, scorer, ks=(5, 10)):
    hits = {k: [] for k in ks}; rr, ndcg = [], []
    for sid in stu[stu.split == split].student_id:
        si = stu.index[stu.student_id == sid][0]
        X = frame(one_vs_all(FM["text"], SM["text"][si]), one_vs_all(FM["title"], SM["title"][si]),
                  one_vs_all(FM["topic"], SM["topic"][si]), FSTAT, stu.n_papers[si])
        P = pairs[(pairs.student_id == sid) & (pairs.label == 1)]
        for _, r in P.iterrows():                        # positives use leakage-free features
            X.loc[fidx[r.faculty_id], FEATURES] = r[FEATURES].values
        c = co[co.student_id == sid]
        masked = {fidx[x] for x in c.faculty_id} - {fidx[x] for x in P.faculty_id}
        s = scorer(X).astype(float)
        s[list(masked)] = -np.inf                        # co-authors we could not score fairly
        order = np.argsort(-s); posi = {fidx[x] for x in P.faculty_id}
        rank_of = {j: r + 1 for r, j in enumerate(order)}
        ranks = sorted(rank_of[j] for j in posi)
        for k in ks: hits[k].append(float(ranks[0] <= k))
        rr.append(1 / ranks[0])
        dcg = sum(1 / np.log2(r + 1) for r in ranks if r <= 10)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ranks), 10)))
        ndcg.append(dcg / idcg)
    out = {f"Hit@{k}": round(np.mean(v), 3) for k, v in hits.items()}
    out.update(MRR=round(np.mean(rr), 3), **{"NDCG@10": round(np.mean(ndcg), 3)})
    return out


print(f"\nRanking over all {len(fac)} faculty, {int((stu.split=='test').sum())} unseen test students:")
rng = np.random.default_rng(0)
cands = [("random", lambda X: rng.random(len(X))),
         ("text similarity only", lambda X: X.sim_text.values)]
cands += [(n, (lambda m, c: lambda X: m.predict_proba(X[c])[:, 1])(m, c)) for n, (m, c) in models.items()]
for split in ("val", "test"):
    print(f"[{split}]")
    for label, fn in cands:
        print(f"  {label:44s}", {k: float(v) for k, v in rank_metrics(split, fn).items()})


# ------------------------------------------------- recommend for free text
def recommend(text, k=5):
    q = {kk: V[kk].transform([text]) for kk in V}
    X = frame(one_vs_all(FM["text"], q["text"]), one_vs_all(FM["title"], q["title"]),
              one_vs_all(FM["topic"], q["topic"]), FSTAT, 2)
    score = model.predict_proba(X[MCOLS])[:, 1]
    an = V["title"].build_analyzer(); qwords = set(an(text))
    out = []
    for rank, j in enumerate(np.argsort(-score)[:k], 1):
        f = fac.iloc[j]; why = []
        common = sorted(qwords & set(an(f.top_topics + " " + f.publications)))
        if common: why.append("Shared keywords: " + ", ".join(common[:6]))
        tops = [t for t in f.top_topics.split("; ")[:3]]
        why.append("Main research areas: " + "; ".join(tops))
        titles = f.publications.split(" | ")
        sims = V["title"].transform(titles) @ q["title"].T
        b = int(np.argmax(sims.toarray().ravel()))
        if sims.toarray().ravel()[b] > 0: why.append(f"Related paper: {titles[b]}")
        why.append(f"{f.n_papers} papers, {f.recent_papers} since 2024, avg {f.avg_citations:.0f} citations")
        out.append({"rank": rank, "faculty": f["name"], "faculty_id": f.faculty_id,
                    "score": round(float(score[j]), 3), "why": why})
    return out


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "machine learning for plant disease detection using image processing"
    print(f"\nQuery: {q}")
    for r in recommend(q):
        print(f"\n#{r['rank']} {r['faculty']} ({r['faculty_id']}) score={r['score']}")
        for w_ in r["why"]: print("   -", w_)
