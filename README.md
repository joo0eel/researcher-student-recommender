#NexusScholar (Name suggestion.............)

# LPU researcher-student collaboration recommender (data + code)

## Files
- dataset/pairs.csv        TRAIN ON THIS. One row per (student, faculty) pair. `label`=1 co-authored, 0 not.
                           Features: sim_text, sim_titles, sim_topics (TF-IDF cosine), fac_n_papers, fac_total_citations,
                           fac_recent_papers, fac_avg_citations, stu_n_papers. `split` = train/val/test (grouped by student).
                           `pair_type` = positive / hard_negative / random_negative.
- dataset/faculty.csv      734 faculty (>=5 papers): stats, top topics, 10 most-cited papers.
- dataset/students.csv     605 anonymised student PROXIES (junior authors with 2-4 papers).
- dataset/coauthorship.csv every junior-senior co-authorship found.
- build_dataset.py         raw OpenAlex CSVs -> dataset/  (re-run if you re-download the data)
- train_recommender.py     trains, evaluates (ranking over all faculty) and recommends from free text
- collab_features.py       shared feature code

## Run
pip install pandas numpy scipy scikit-learn joblib
python build_dataset.py lpu_faculty.csv lpu_works.csv dataset
python train_recommender.py "your interests here"

## Method notes (for the report)
- Students are proxies: junior authors. Label = co-authorship with a senior author.
- Leakage control: for positive pairs the shared paper(s) are removed from BOTH profiles before computing similarity.
- Negatives: 1 hard (topically similar, never co-authored) + 2 random per positive; negatives may include good matches that never happened.
- Split by student, so test students are unseen. Ranking is evaluated against ALL faculty.
