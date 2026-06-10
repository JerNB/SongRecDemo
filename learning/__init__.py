"""P4: shadow learned-ranking training closure.

This package builds the first learned-ranking training loop on top of the P3
feedback logs WITHOUT replacing the P0 rule ranker:

    dataset.py      -- turn FeedbackStore logs into (X, y, sample_weight).
    ranker.py       -- a lightweight, explainable LearnedRanker (logistic
                       regression by default) that outputs a learned_score
                       in [0, 1].
    train_ranker.py -- CLI that reads feedback.sqlite, trains, and saves the
                       model + feature schema.

Nothing here drives the live ordering: the learned score is produced in the
background (shadow mode) for analysis and future, gradual takeover.
"""

from __future__ import annotations

from .dataset import (
    FEATURE_SOURCE_FLAGS,
    LABEL_RULES,
    PICK_TYPES,
    TrainingData,
    build_training_data,
    feature_dict_from_card,
    feature_names_for,
    feature_vector,
    extract_feature_dict,
    label_for_events,
)
from .ranker import LearnedRanker

__all__ = [
    "TrainingData",
    "build_training_data",
    "extract_feature_dict",
    "feature_dict_from_card",
    "feature_names_for",
    "feature_vector",
    "label_for_events",
    "LABEL_RULES",
    "PICK_TYPES",
    "FEATURE_SOURCE_FLAGS",
    "LearnedRanker",
]
