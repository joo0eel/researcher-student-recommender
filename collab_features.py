"""Shared feature definitions so the training data and live recommendations use identical features."""
import numpy as np
import pandas as pd

FEATURES = ["sim_text", "sim_titles", "sim_topics",
            "fac_n_papers", "fac_total_citations", "fac_recent_papers", "fac_avg_citations",
            "stu_n_papers"]


def frame(sim_text, sim_titles, sim_topics, fac_stats, stu_n):
    """fac_stats: DataFrame with n_papers,total_citations,recent_papers,avg_citations (aligned rows)."""
    return pd.DataFrame({
        "sim_text": sim_text, "sim_titles": sim_titles, "sim_topics": sim_topics,
        "fac_n_papers": np.asarray(fac_stats["n_papers"]),
        "fac_total_citations": np.asarray(fac_stats["total_citations"]),
        "fac_recent_papers": np.asarray(fac_stats["recent_papers"]),
        "fac_avg_citations": np.asarray(fac_stats["avg_citations"]),
        "stu_n_papers": stu_n})[FEATURES]


def one_vs_all(F, q):
    """Cosine of one L2-normalised row vector q against every row of F (also L2-normalised)."""
    return (F @ q.T).toarray().ravel()


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class NonNegativeScorer(BaseEstimator, ClassifierMixin):
    """Linear scorer whose weights are all >= 0, so more similarity can never lower a score
    (plain logistic regression gave a negative weight to one correlated similarity feature)."""
    def fit(self, X, y):
        self.sc_ = StandardScaler().fit(X)
        self.lin_ = LinearRegression(positive=True).fit(self.sc_.transform(X), y)
        self.coef_ = self.lin_.coef_
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        s = np.clip(self.lin_.predict(self.sc_.transform(X)), 0, 1)
        return np.column_stack([1 - s, s])
