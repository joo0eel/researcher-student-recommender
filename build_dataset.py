"""
Build a project-ready dataset from the two OpenAlex CSVs.

  python build_dataset.py lpu_faculty.csv lpu_works.csv dataset

Outputs (in ./dataset):
  faculty.csv      cleaned faculty (>=5 papers) with stats, topics, top papers
  students.csv     anonymised 'student proxies' (junior authors, 2-4 papers) + train/val/test split
  pairs.csv        labelled student-faculty pairs with numeric features  <- train on this
  coauthorship.csv every junior-senior co-authorship found
  artifacts/       TF-IDF vectorizers + profile vectors used by train_recommender.py
Label = 1 if the junior and senior author co-authored >=1 paper.
LEAKAGE FIX: for positive pairs, the shared paper(s) are removed from BOTH profiles before similarity is computed.
"""
import os, sys, pickle
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from collab_features import frame, FEATURES

FAC_MIN, STU_MIN, STU_MAX = 5, 2, 4
rng = np.random.default_rng(42)


def ssum(M, idx):
    return sparse.csr_matrix(np.ones(len(idx))) @ M[idx]


def prof(M, idx, exclude=frozenset()):
    keep = [i for i in idx if i not in exclude]
    if not keep:
        return sparse.csr_matrix((1, M.shape[1]))
    return normalize(ssum(M, keep))


def cos(a, b):
    return float(a.multiply(b).sum())


def top_topics(idx, topics, k=8):
    c = Counter(t.strip() for i in idx for t in topics[i].split(";") if t.strip())
    return "; ".join(t for t, _ in c.most_common(k))


def main(fac_csv, works_csv, out="dataset"):
    os.makedirs(out + "/artifacts", exist_ok=True)
    f = pd.read_csv(fac_csv).dropna(subset=["faculty_id"]).copy()
    w = pd.read_csv(works_csv)
    w["title"] = w.title.fillna("")
    w["key"] = w.title.str.lower().str.strip()
    w = w[w.key != ""].drop_duplicates("key").reset_index(drop=True)
    w["topics"] = w.topics.fillna(""); w["abstract"] = w.abstract.fillna("")
    topic_txt = w.topics.str.replace(";", " ", regex=False)
    w["doc"] = w.title + ". " + topic_txt + ". " + w.abstract
    key2i = {k: i for i, k in enumerate(w.key)}
    topics = w.topics.tolist()

    ap, names = {}, {}
    for fid, nm, pubs in zip(f.faculty_id.astype(str), f.name, f.publications.fillna("")):
        idx = sorted({key2i[t.strip().lower()] for t in pubs.split(" | ") if t.strip().lower() in key2i})
        if idx:
            ap[fid] = idx; names[fid] = nm
    n = {a: len(i) for a, i in ap.items()}
    print(f"authors with matched papers: {len(ap)}")

    tf_text = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5, sublinear_tf=True,
                              ngram_range=(1, 2), max_features=40000).fit(w.doc)
    tf_title = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit(w.title)
    tf_topic = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True).fit(topic_txt)
    PM = {"text": tf_text.transform(w.doc), "title": tf_title.transform(w.title),
          "topic": tf_topic.transform(topic_txt)}
    K = ["text", "title", "topic"]

    fac_ids = [a for a in ap if n[a] >= FAC_MIN]
    stu_ids = [a for a in ap if STU_MIN <= n[a] <= STU_MAX]
    p2f = defaultdict(set)
    for a in fac_ids:
        for p in ap[a]:
            p2f[p].add(a)

    # ---- co-authorship (positives)
    shared, by_stu = {}, defaultdict(dict)
    for s in stu_ids:
        cnt = defaultdict(set)
        for p in ap[s]:
            for fa in p2f.get(p, ()):
                cnt[fa].add(p)
        for fa, sh in cnt.items():
            shared[(s, fa)] = sh; by_stu[s][fa] = sh
    pos = {k: v for k, v in shared.items() if len(v) < n[k[0]]}      # student must keep >=1 paper
    stu_used = sorted({s for s, _ in pos})
    print(f"faculty: {len(fac_ids)} | student proxies with a senior co-author: {len(stu_used)} | positive pairs: {len(pos)}")

    # ---- faculty table + stats
    fidx = {a: i for i, a in enumerate(fac_ids)}
    sidx = {a: i for i, a in enumerate(stu_used)}
    fcode = {a: f"F{i:04d}" for a, i in fidx.items()}
    scode = {a: f"S{i:04d}" for a, i in sidx.items()}
    cit, yr = w.cited_by.values, w.year.values
    fs = pd.DataFrame({
        "faculty_id": [fcode[a] for a in fac_ids], "openalex_id": fac_ids,
        "name": [names[a] for a in fac_ids], "n_papers": [n[a] for a in fac_ids],
        "total_citations": [int(cit[ap[a]].sum()) for a in fac_ids],
        "recent_papers": [int((yr[ap[a]] >= 2024).sum()) for a in fac_ids]})
    fs["avg_citations"] = (fs.total_citations / fs.n_papers).round(2)
    fs["top_topics"] = [top_topics(ap[a], topics) for a in fac_ids]
    fs["publications"] = [" | ".join(w.title.iloc[sorted(ap[a], key=lambda i: -cit[i])[:10]]) for a in fac_ids]

    # ---- full profiles
    FP = {k: normalize(sparse.vstack([prof(PM[k], ap[a]) for a in fac_ids]).tocsr()) for k in K}
    SP = {k: normalize(sparse.vstack([prof(PM[k], ap[a]) for a in stu_used]).tocsr()) for k in K}
    sim_all = (SP["text"] @ FP["text"].T).toarray()

    # ---- split by student
    order = rng.permutation(len(stu_used))
    split = np.empty(len(stu_used), dtype=object)
    split[order[:int(.70 * len(order))]] = "train"
    split[order[int(.70 * len(order)):int(.85 * len(order))]] = "val"
    split[order[int(.85 * len(order)):]] = "test"

    # ---- pair list: positives + hard + random negatives
    pairs = []
    for s in stu_used:
        pf = [fa for (ss, fa) in pos if ss == s]
        blocked = set(by_stu[s])                                    # any co-author: not a valid negative
        for fa in pf:
            pairs.append((s, fa, 1, "positive"))
        cand = [j for j in np.argsort(-sim_all[sidx[s]]) if fac_ids[j] not in blocked]
        hard = rng.choice(cand[:15], size=min(len(pf), 15), replace=False)
        rest = [j for j in cand[15:]]
        rand = rng.choice(rest, size=min(2 * len(pf), len(rest)), replace=False)
        pairs += [(s, fac_ids[j], 0, "hard_negative") for j in hard]
        pairs += [(s, fac_ids[j], 0, "random_negative") for j in rand]

    rows = []
    for s, fa, y, typ in pairs:
        si, fi = sidx[s], fidx[fa]
        if y:
            sh = pos[(s, fa)]
            sv = {k: prof(PM[k], ap[s], sh) for k in K}
            fv = {k: prof(PM[k], ap[fa], sh) for k in K}
        else:
            sv = {k: SP[k][si] for k in K}; fv = {k: FP[k][fi] for k in K}
        rows.append((cos(sv["text"], fv["text"]), cos(sv["title"], fv["title"]),
                     cos(sv["topic"], fv["topic"])))
    sims = np.array(rows)
    fstat = fs.iloc[[fidx[fa] for _, fa, _, _ in pairs]].reset_index(drop=True)
    X = frame(sims[:, 0], sims[:, 1], sims[:, 2], fstat, [n[s] for s, *_ in pairs])
    pr = pd.concat([pd.DataFrame({"student_id": [scode[s] for s, *_ in pairs],
                                  "faculty_id": [fcode[fa] for _, fa, *_ in pairs],
                                  "label": [y for *_, y, _ in pairs],
                                  "pair_type": [t for *_, t in pairs]}), X], axis=1)
    pr["split"] = pr.student_id.map(dict(zip([scode[a] for a in stu_used], split)))

    st = pd.DataFrame({"student_id": [scode[a] for a in stu_used], "openalex_id": stu_used,
                       "n_papers": [n[a] for a in stu_used],
                       "top_topics": [top_topics(ap[a], topics) for a in stu_used],
                       "publications": [" | ".join(w.title.iloc[ap[a]][:6]) for a in stu_used],
                       "split": split})
    co = pd.DataFrame([(scode[s], fcode[fa], len(sh)) for (s, fa), sh in shared.items() if s in sidx],
                      columns=["student_id", "faculty_id", "n_shared_papers"])

    fs.to_csv(f"{out}/faculty.csv", index=False)
    st.to_csv(f"{out}/students.csv", index=False)
    pr.to_csv(f"{out}/pairs.csv", index=False)
    co.to_csv(f"{out}/coauthorship.csv", index=False)
    with open(f"{out}/artifacts/vectorizers.pkl", "wb") as fh:
        pickle.dump({"text": tf_text, "title": tf_title, "topic": tf_topic}, fh)
    for k in K:
        sparse.save_npz(f"{out}/artifacts/fac_{k}.npz", FP[k])
        sparse.save_npz(f"{out}/artifacts/stu_{k}.npz", SP[k])
    print(pr.label.value_counts().to_dict(), "| pairs:", len(pr))
    print(pr.groupby("label")[["sim_text", "sim_titles", "sim_topics"]].mean().round(3))
    print(pr.groupby("split").student_id.nunique().to_dict())


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "lpu_faculty.csv", a[1] if len(a) > 1 else "lpu_works.csv",
         a[2] if len(a) > 2 else "dataset")
