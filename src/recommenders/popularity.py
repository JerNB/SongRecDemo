"""
Popularity-Based Recommender (Baseline).

Design
------
This is the simplest possible recommender and serves as a lower-bound
baseline.  Every user receives the same ranked list: the globally most
popular tracks in the training data.

Why this is a meaningful baseline (not just filler)
----------------------------------------------------
- It is surprisingly hard to beat for Precision@K in highly skewed
  popularity distributions (Gini ≈ 0.59 for this dataset).
- Comparing CF / content models against it reveals whether they
  add value beyond "recommend the hits".
- Popularity is also used as the ``novelty`` denominator in evaluation:
  items popular in training are considered less "novel".

Scoring options (controlled via config.POPULARITY_SCORE)
---------------------------------------------------------
- "count"     : score = raw interaction count in training
- "log_count" : score = log(1 + count); dampens superstar effect slightly

Both options produce the same ranking if no items are tied; log-count
gives a smoother score distribution for diversity analysis.

Cold-start behaviour
--------------------
Because every user gets the same list (minus seen items), this model
handles new users perfectly.  It is blind to user preferences, which
is its main limitation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
from src.recommenders.base import BaseRecommender

log = logging.getLogger(__name__)


class PopularityRecommender(BaseRecommender):
    """Recommend globally popular tracks from the training period."""

    def __init__(self, score_mode: str = config.POPULARITY_SCORE) -> None:
        if score_mode not in ("count", "log_count"):
            raise ValueError(f"Unknown score_mode: {score_mode!r}")
        self._score_mode = score_mode
        self._popularity: pd.Series | None = None        # item_id_raw -> score
        # Cached numpy views for fast recommend() (set at fit time)
        self._pop_ids: np.ndarray | None = None
        self._pop_scores: np.ndarray | None = None
        self._user_seen: dict[str, set[str]] | None = None

    @property
    def name(self) -> str:
        return f"Popularity({self._score_mode})"

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> "PopularityRecommender":
        """Count interactions per item in the training split.

        Scores are derived from TRAINING data only; computing them on
        the full dataset would leak test-set information into the ranking.
        """
        counts = train_df["item_id_raw"].value_counts()
        if self._score_mode == "log_count":
            scores = np.log1p(counts)
        else:
            scores = counts.astype(float)

        # Deterministic ordering: primary by score desc, secondary by
        # item_id_raw asc.  Without the secondary key, items tied on
        # popularity would be ranked in pandas-internal (hash) order,
        # making the top-K list non-reproducible across runs / pandas
        # versions.  Ties are common at the long tail of the popularity
        # distribution, so this matters even though top-K on the short
        # head is stable.
        ordering = pd.DataFrame(
            {"score": scores.values, "item_id_raw": scores.index}
        ).sort_values(
            by=["score", "item_id_raw"],
            ascending=[False, True],
            kind="mergesort",
        )
        self._popularity = pd.Series(
            ordering["score"].to_numpy(),
            index=ordering["item_id_raw"].to_numpy(),
            name="popularity",
        )
        # Pre-materialise numpy views so recommend() avoids per-user copies
        self._pop_ids = self._popularity.index.to_numpy()
        self._pop_scores = self._popularity.to_numpy()

        # Cache each user's training interactions for exclusion at inference
        self._user_seen = (
            train_df.groupby("user_id_raw")["item_id_raw"]
            .apply(set)
            .to_dict()
        )

        log.info(
            "PopularityRecommender fitted: %d items scored | "
            "top item=%s (score=%.1f)",
            len(self._popularity),
            self._popularity.index[0],
            self._popularity.iloc[0],
        )
        return self

    # ------------------------------------------------------------------
    # Recommend
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id_raw: str,
        n: int,
        exclude_seen: bool = True,
    ) -> list[tuple[str, float]]:
        if self._popularity is None or self._pop_ids is None:
            raise RuntimeError("Call fit() before recommend().")

        seen = (
            self._user_seen.get(user_id_raw, set())
            if exclude_seen and self._user_seen is not None
            else set()
        )

        # Walk the pre-sorted popularity list in order and skip seen items.
        # Worst case O(n_items); typically stops after ~n + |seen ∩ head|
        # steps.  Faster than copying + dropping per user.
        top: list[tuple[str, float]] = []
        for iid, score in zip(self._pop_ids, self._pop_scores):
            if iid in seen:
                continue
            top.append((iid, float(score)))
            if len(top) >= n:
                break
        return top

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain(self, user_id_raw: str, item_id_raw: str) -> str:
        if self._popularity is None:
            return "Model not yet fitted."
        score = self._popularity.get(item_id_raw, 0)
        if self._score_mode == "count":
            return (
                f"This track was interacted with by {int(score)} users in the "
                "training data, placing it among the most popular items in the catalogue."
            )
        return (
            f"This track has a log-popularity score of {score:.2f} (raw count "
            f"{int(np.expm1(score))}), making it one of the most listened-to items."
        )
