"""
Evaluation metrics for offline implicit-feedback recommendation.

All metric functions are pure (no side-effects) and operate on simple
Python types so they can be unit-tested independently of any model.

Metric taxonomy used in this project
-------------------------------------
Accuracy metrics (relevance):
  - Precision@K   : fraction of recommended items that are relevant
  - Recall@K      : fraction of relevant items that were recommended
  - F1@K          : harmonic mean of precision and recall at K
  - NDCG@K        : normalised discounted cumulative gain (rank-weighted)
  - Hit Rate@K    : 1 if ≥1 relevant item appears in top-K (binary)

Beyond-accuracy metrics (quality of discovery):
  - Catalog Coverage@K  : fraction of the item catalogue recommended
                          to at least one user (across all users)
  - Intra-List Diversity : average pairwise cosine distance within each
                           recommendation list (requires item feature matrix)
  - Novelty@K           : average popularity-adjusted surprise; lower
                           popularity → higher novelty score

Definitions
-----------
"Relevant" in offline evaluation = appears in the user's held-out
(validation or test) interaction set.  Because interactions are binary
implicit feedback (no graded ratings), all held-out items are treated
as equally relevant.

Notation
--------
K           : recommendation list length
n_users     : number of evaluated users
catalogue   : set of all item_ids in the training catalogue
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Per-user accuracy helpers
# ---------------------------------------------------------------------------

def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-K recommended items that are relevant.

    Parameters
    ----------
    recommended : list[str]
        Ordered item_ids (best first), length may exceed K.
    relevant : set[str]
        Ground-truth held-out items for this user.
    k : int
        Cutoff.

    Returns
    -------
    float in [0, 1].  Returns 0.0 when recommended or relevant is empty.
    """
    if not recommended or not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items that appear in the top-K list.

    Returns 0.0 when relevant is empty (undefined; treated as 0).
    """
    if not recommended or not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def f1_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at K.

    Uses binary relevance: gain = 1 if item is relevant, 0 otherwise.
    Ideal DCG (IDCG) is computed as if the top min(|relevant|, K)
    positions are all relevant.

    Returns 0.0 when relevant is empty.
    """
    if not recommended or not relevant:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """1.0 if at least one relevant item appears in top-K, else 0.0."""
    top_k = set(recommended[:k])
    return 1.0 if top_k & relevant else 0.0


# ---------------------------------------------------------------------------
# Aggregation helpers (over all users)
# ---------------------------------------------------------------------------

def mean_metric(
    per_user_values: list[float],
    exclude_nan: bool = True,
) -> float:
    """Arithmetic mean, optionally ignoring NaN entries."""
    if not per_user_values:
        return float("nan")
    if exclude_nan:
        vals = [v for v in per_user_values if not math.isnan(v)]
    else:
        vals = per_user_values
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Beyond-accuracy metrics
# ---------------------------------------------------------------------------

def catalog_coverage(
    recommendations: dict[str, list[str]],
    catalogue: set[str],
    k: int,
) -> float:
    """Fraction of the catalogue recommended to at least one user.

    Parameters
    ----------
    recommendations : dict[user_id_raw -> list[item_id_raw]]
        Top-K lists for every evaluated user (order matters for @K cutoff).
    catalogue : set[str]
        All item_ids known at training time.
    k : int
        Cutoff applied to each user's list.

    Returns
    -------
    float in [0, 1].

    Interpretation
    --------------
    A popularity-only baseline will have very low coverage (same ~K items
    recommended to everyone).  A diverse model should push this higher.
    Coverage of 1.0 means every catalogue item was recommended to at least
    one user.
    """
    if not catalogue:
        return float("nan")
    recommended_items: set[str] = set()
    for recs in recommendations.values():
        recommended_items.update(recs[:k])
    return len(recommended_items & catalogue) / len(catalogue)


def intra_list_diversity(
    recommendations: dict[str, list[str]],
    item_vectors: sp.csr_matrix,
    item_index: list[str],
    k: int,
) -> float:
    """Mean pairwise cosine distance within each recommendation list.

    Parameters
    ----------
    recommendations : dict[user_id_raw -> list[item_id_raw]]
    item_vectors : sp.csr_matrix, shape (n_items, n_features)
        L2-normalised item feature matrix (cosine sim = dot product).
    item_index : list[str]
        Ordered item_id_raw values corresponding to rows of item_vectors.
    k : int

    Returns
    -------
    float in [0, 1].  0 means all recommended items are identical in
    feature space; 1 means all pairs are orthogonal.

    Notes
    -----
    Cosine distance = 1 - cosine_similarity.  For L2-normalised vectors,
    cosine_similarity(a, b) = a · b, so distance = 1 - a · b.
    """
    item_pos = {iid: i for i, iid in enumerate(item_index)}
    per_user_diversities: list[float] = []

    for recs in recommendations.values():
        top_k = [r for r in recs[:k] if r in item_pos]
        if len(top_k) < 2:
            continue

        indices = [item_pos[iid] for iid in top_k]
        vecs = np.asarray(item_vectors[indices].todense())   # shape (|top_k|, d)

        # Pairwise cosine similarities via matrix multiply
        sim_matrix = vecs @ vecs.T   # (n × n)
        n = len(top_k)
        # Upper-triangle (excluding diagonal) pairs
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += 1.0 - sim_matrix[i, j]
                count += 1

        if count > 0:
            per_user_diversities.append(total_dist / count)

    return float(np.mean(per_user_diversities)) if per_user_diversities else float("nan")


def novelty_at_k(
    recommendations: dict[str, list[str]],
    item_popularity: dict[str, int],
    n_users: int,
    k: int,
) -> float:
    """Mean self-information novelty (popularity-adjusted surprise) at K.

    Novelty(i) = -log2(popularity(i) / n_users)

    A track listened to by every user has novelty ≈ 0 (no surprise).
    A track listened to by very few users has high novelty.

    Parameters
    ----------
    recommendations : dict[user_id_raw -> list[item_id_raw]]
    item_popularity : dict[item_id_raw -> int]
        Number of training-set users who interacted with each item.
    n_users : int
        Total number of users in the training set.
    k : int

    Returns
    -------
    float  (higher = more novel/surprising recommendations on average)
    """
    novelty_scores: list[float] = []
    for recs in recommendations.values():
        for item in recs[:k]:
            pop = item_popularity.get(item, 1)   # floor at 1 to avoid log(0)
            novelty_scores.append(-math.log2(pop / n_users))
    return float(np.mean(novelty_scores)) if novelty_scores else float("nan")


# ---------------------------------------------------------------------------
# Full metric suite (called by evaluator)
# ---------------------------------------------------------------------------

def compute_all_metrics(
    recommendations: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    catalogue: set[str],
    k_values: list[int],
    item_popularity: Optional[dict[str, int]] = None,
    item_vectors: Optional[sp.csr_matrix] = None,
    item_index: Optional[list[str]] = None,
    n_train_users: int = 1,
) -> dict[str, float]:
    """Compute all metrics for one model at all configured K values.

    Parameters
    ----------
    recommendations : dict[user_id_raw -> list[item_id_raw]]
        Ranked lists produced by the model for each evaluation user.
    ground_truth : dict[user_id_raw -> set[item_id_raw]]
        Held-out items per user.
    catalogue : set[str]
        All item_ids seen during training.
    k_values : list[int]
        e.g. [5, 10, 20]
    item_popularity : dict[str, int], optional
        Required for Novelty.
    item_vectors : sp.csr_matrix, optional
        Required for Intra-List Diversity.
    item_index : list[str], optional
        Required when item_vectors is provided.
    n_train_users : int
        Required for Novelty denominator.

    Returns
    -------
    dict[metric_name -> float]
        All metric values in a flat dict, e.g.
        ``{"precision@10": 0.123, "recall@10": 0.087, ...}``
    """
    results: dict[str, float] = {}

    for k in k_values:
        p_vals, r_vals, f1_vals, ndcg_vals, hr_vals = [], [], [], [], []

        for uid, recs in recommendations.items():
            relevant = ground_truth.get(uid, set())
            if not relevant:
                continue
            p_vals.append(precision_at_k(recs, relevant, k))
            r_vals.append(recall_at_k(recs, relevant, k))
            f1_vals.append(f1_at_k(recs, relevant, k))
            ndcg_vals.append(ndcg_at_k(recs, relevant, k))
            hr_vals.append(hit_rate_at_k(recs, relevant, k))

        results[f"precision@{k}"] = mean_metric(p_vals)
        results[f"recall@{k}"] = mean_metric(r_vals)
        results[f"f1@{k}"] = mean_metric(f1_vals)
        results[f"ndcg@{k}"] = mean_metric(ndcg_vals)
        results[f"hit_rate@{k}"] = mean_metric(hr_vals)
        results[f"coverage@{k}"] = catalog_coverage(recommendations, catalogue, k)

        if item_popularity is not None:
            results[f"novelty@{k}"] = novelty_at_k(
                recommendations, item_popularity, n_train_users, k
            )

        if item_vectors is not None and item_index is not None:
            results[f"diversity@{k}"] = intra_list_diversity(
                recommendations, item_vectors, item_index, k
            )

    return results
