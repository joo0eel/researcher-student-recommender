"""
Feature definitions and helper functions for supervised student-faculty collaboration prediction.

This module defines:
- The numeric feature sets used by the supervised classifiers.
- Utility functions to construct feature frames and compute similarity scores.
"""

from typing import Union, Sequence
import numpy as np
import pandas as pd
from scipy import sparse


# Feature column definitions
SIMILARITY_FEATURES = ["sim_text", "sim_titles", "sim_topics"]
FACULTY_FEATURES = [
    "fac_n_papers",
    "fac_total_citations",
    "fac_recent_papers",
    "fac_avg_citations",
]
STUDENT_FEATURES = ["stu_n_papers"]

# Full feature list used by supervised machine learning models
FEATURES = SIMILARITY_FEATURES + FACULTY_FEATURES + STUDENT_FEATURES


def build_feature_frame(
    sim_text: Union[np.ndarray, Sequence[float]],
    sim_titles: Union[np.ndarray, Sequence[float]],
    sim_topics: Union[np.ndarray, Sequence[float]],
    faculty_stats: pd.DataFrame,
    student_paper_count: Union[int, np.ndarray, Sequence[int]],
) -> pd.DataFrame:
    """
    Constructs a standardized feature DataFrame matching the supervised model's expected inputs.

    Parameters:
    -----------
    sim_text : array-like
        Cosine similarity between student and faculty full text profiles.
    sim_titles : array-like
        Cosine similarity between student and faculty paper titles.
    sim_topics : array-like
        Cosine similarity between student and faculty topic keywords.
    faculty_stats : pd.DataFrame
        DataFrame containing columns: ['n_papers', 'total_citations', 'recent_papers', 'avg_citations'].
    student_paper_count : int or array-like
        Number of papers published by the student.

    Returns:
    --------
    pd.DataFrame containing the ordered features matching `FEATURES`.
    """
    return pd.DataFrame(
        {
            "sim_text": np.asarray(sim_text),
            "sim_titles": np.asarray(sim_titles),
            "sim_topics": np.asarray(sim_topics),
            "fac_n_papers": np.asarray(faculty_stats["n_papers"]),
            "fac_total_citations": np.asarray(faculty_stats["total_citations"]),
            "fac_recent_papers": np.asarray(faculty_stats["recent_papers"]),
            "fac_avg_citations": np.asarray(faculty_stats["avg_citations"]),
            "stu_n_papers": student_paper_count,
        }
    )[FEATURES]


def compute_cosine_similarities(matrix: sparse.csr_matrix, query_vector: sparse.csr_matrix) -> np.ndarray:
    """
    Computes cosine similarity between an L2-normalized query row vector and all rows of a matrix.

    Parameters:
    -----------
    matrix : sparse.csr_matrix (N x V)
        L2-normalized sparse matrix of faculty profile vectors.
    query_vector : sparse.csr_matrix (1 x V)
        L2-normalized sparse vector representing the student query profile.

    Returns:
    --------
    np.ndarray of shape (N,) containing cosine similarity scores in range [0, 1].
    """
    return (matrix @ query_vector.T).toarray().ravel()


# Backward-compatibility aliases
frame = build_feature_frame
one_vs_all = compute_cosine_similarities
