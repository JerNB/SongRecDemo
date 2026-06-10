"""Offline evaluation harness (P3).

A small, dependency-light harness that runs the recommender against a fixed
set of seed profiles and reports *diagnostic* metrics (coverage, diversity,
novelty, source mix, embedding share, duplicate rate, latency, score
distribution).

These are deliberately NOT accuracy metrics: there are no human relevance
labels yet, so the harness never claims Precision / Recall / NDCG. Drop a
``manual_labels.json`` next to ``seed_profiles.json`` later to unlock those.
"""

from __future__ import annotations

from .metrics import (
    compute_all_metrics,
    coverage_at_k,
    diversity_at_k,
    duplicate_rate_at_k,
    embedding_share_at_k,
    learned_shadow_metrics,
    novelty_at_k,
    score_distribution,
    source_mix_at_k,
)
from .profiles import SeedProfile, load_seed_profiles

__all__ = [
    "SeedProfile",
    "load_seed_profiles",
    "compute_all_metrics",
    "coverage_at_k",
    "diversity_at_k",
    "novelty_at_k",
    "source_mix_at_k",
    "embedding_share_at_k",
    "duplicate_rate_at_k",
    "score_distribution",
    "learned_shadow_metrics",
]
