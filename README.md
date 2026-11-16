# NexusScholar: Supervised Researcher-Student Collaboration Prediction & Recommender

A supervised machine learning system that predicts academic research collaborations between students and faculty members, and recommends relevant faculty advisors based on learned collaboration probabilities.

---

## 1. Supervised Machine Learning Formulation

This project is framed as a **Supervised Binary Classification & Link Prediction** problem:

$$\hat{y} = P(Y = 1 \mid \mathbf{x})$$

- **Input ($\mathbf{x}$)**: An 8-dimensional feature vector capturing content similarity and researcher profile statistics between a student and a faculty member.
- **Target ($Y$)**: Binary ground-truth label:
  - `1`: **Collaboration / Co-authored** (student and faculty co-authored $\ge 1$ paper).
  - `0`: **Non-collaboration / Negative pair** (hard negatives with similar topics or random faculty).
- **Supervised Splits**: Stratified and grouped by student so that test students are completely unseen during training:
  - **Train**: 4,340 pairs (70%)
  - **Validation**: 880 pairs (15%)
  - **Test**: 990 pairs (15%)

---

## 2. Labeled Dataset (`dataset/`)

All raw, unlabeled datasets and web scrapers have been removed. The project directly uses the cleaned, verified labeled dataset:

| File | Description |
| :--- | :--- |
| **`dataset/pairs.csv`** | **The Primary Labeled Dataset** (6,210 rows). Contains all 8 features, ground-truth `label` ($0$ or $1$), pair type (`positive`, `hard_negative`, `random_negative`), and `split` (`train`, `val`, `test`). |
| **`dataset/faculty.csv`** | 734 faculty members ($\ge 5$ papers) with metadata: names, publication counts, citation statistics, top research topics, and key publications. |
| **`dataset/students.csv`** | 605 student profiles (junior researchers with 2–4 papers) mapped to train/val/test splits. |
| **`dataset/coauthorship.csv`** | Ground-truth co-authorship records between students and faculty. |
| **`dataset/artifacts/`** | Pre-computed TF-IDF vectorizers (`vectorizers.pkl`) and L2-normalized sparse profile matrices (`fac_text.npz`, `fac_title.npz`, `fac_topic.npz`, etc.). |
| **`dataset/model.joblib`** | Serialized best-performing supervised model and feature list. |

### Feature Descriptions ($\mathbf{x}$)

1. **`sim_text`**: Cosine similarity between student and faculty full text profiles (abstracts + titles + topics).
2. **`sim_titles`**: Cosine similarity between student and faculty publication titles.
3. **`sim_topics`**: Cosine similarity between student and faculty topic keywords.
4. **`fac_n_papers`**: Total papers published by the faculty member.
5. **`fac_total_citations`**: Total citations received by the faculty member.
6. **`fac_recent_papers`**: Faculty papers published since 2024 (recency indicator).
7. **`fac_avg_citations`**: Average citations per paper for the faculty member.
8. **`stu_n_papers`**: Number of papers published by the student.

---

## 3. Supervised Classifiers & Evaluation

The pipeline trains and benchmarks three supervised classification algorithms:

1. **Logistic Regression (Standardized)**: Linear probabilistic classifier with StandardScaler and balanced class weights.
2. **Random Forest Classifier**: Non-linear tree ensemble with balanced class weights.
3. **Gradient Boosting Classifier**: Sequential boosting trees optimizing log-loss.

### Benchmark Results on Held-out Test Set (990 pairs):

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 64.9% | 0.357 | 0.496 | 0.415 | 0.661 |
| **Random Forest** | 64.4% | 0.388 | **0.730** | **0.507** | 0.724 |
| **Gradient Boosting** | **76.5%** | **0.586** | 0.206 | 0.305 | **0.739** |

### Feature Importance (Gradient Boosting):
- `sim_topics` (27.3%): Topic match is the strongest predictor of collaboration.
- `sim_text` (23.1%): In-depth profile textual alignment.
- `sim_titles` (20.4%): Specific publication title keywords.
- `fac_total_citations` (7.5%), `fac_avg_citations` (7.2%), `fac_n_papers` (7.2%): Faculty experience and research productivity.
- `fac_recent_papers` (5.0%), `stu_n_papers` (2.3%): Recency and student publication activity.

---

## 4. Project Structure

```
research ML/
│
├── dataset/
│   ├── pairs.csv              # Labeled dataset with 8 features, labels (0/1), and splits
│   ├── faculty.csv            # 734 faculty profiles (names, citations, topics, papers)
│   ├── students.csv           # 605 student profiles and split assignments
│   ├── coauthorship.csv       # Ground-truth co-authorship relationships
│   ├── model.joblib           # Trained supervised classification model
│   └── artifacts/             # TF-IDF vectorizers and profile sparse matrices
│
├── collab_features.py         # Feature definitions and similarity utilities
├── train_recommender.py       # Supervised model training, evaluation & ranking
├── recommend.py               # Interactive / CLI recommendation tool
├── README.md                  # Project documentation
└── .gitignore                 # Git ignore rules
```

---

## 5. Quick Start & Usage

### Prerequisites
Install the required Python packages:
```bash
pip install numpy pandas scipy scikit-learn joblib
```

### Train and Evaluate Supervised Models
To train the supervised classifiers, evaluate performance on test data, and run ranking metrics:
```bash
python train_recommender.py
```

You can also pass a query directly to train and recommend in one step:
```bash
python train_recommender.py "deep learning for plant disease detection using image processing"
```

### Recommend Faculty for a Student Query
Use `recommend.py` to get instant recommendations using the pre-trained supervised model without re-training:
```bash
python recommend.py "natural language processing and large language models"
```
Or run interactively:
```bash
python recommend.py
```

Example Output:
```
Searching faculty matches for query: "machine learning for plant disease detection using image processing"

----------------------------------------------------------------------
Rank #1: P. B. Narendra Kiran (F0556)
Predicted Collaboration Probability: 84.7%
   * Shared keywords: detection, learning, machine, using
   * Main research areas: AI and HR Technologies; Financial Distress and Bankruptcy Prediction
   * Relevant paper: Novel Method for Detection of Stress in Employees using Hybrid Deep learning Models
   * Profile: 6 papers, 3 since 2024, avg 22.7 citations
----------------------------------------------------------------------
Rank #2: Shabnam Shabir (F0315)
Predicted Collaboration Probability: 81.8%
   * Shared keywords: disease, plant
   * Main research areas: Parkinson's Disease Mechanisms and Treatments; Phytochemicals
   * Profile: 11 papers, 1 since 2024, avg 19.1 citations
----------------------------------------------------------------------
Rank #3: Veerpal Kaur (F0568)
Predicted Collaboration Probability: 81.7%
   * Shared keywords: detection, image, learning, machine, using
   * Main research areas: Artificial Intelligence in Healthcare; AI in cancer detection
   * Relevant paper: Cancer Detection and Analysis Using Machine Learning
   * Profile: 6 papers, 2 since 2024, avg 13.8 citations
----------------------------------------------------------------------
```
