"""Score-level hybrid of implicit ALS and tag TF-IDF recommenders.

The two component models produce scores on incompatible scales: ALS dot
products are unbounded and may be negative, while TF-IDF cosine similarity is
bounded and non-negative.  This class therefore performs *per-user min-max
normalisation over the unseen catalogue* before blending:

    hybrid(u, i) = w * als_norm(u, i) + (1 - w) * tfidf_norm(u, i)

Normalising per user preserves the ordering and relative score gaps of each
component while preventing either model from winning merely because its raw
numbers have a larger numerical range.  All catalogue items are scored; there
is no candidate-union approximation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.recommenders.base import BaseRecommender
from src.recommenders.collaborative import CollaborativeFilteringRecommender
from src.recommenders.content_based import ContentBasedRecommender

log = logging.getLogger(__name__)


class HybridRecommender(BaseRecommender):
    """Blend fitted ALS and TF-IDF scores after per-user normalisation."""

    def __init__(
        self,
        als_model: CollaborativeFilteringRecommender,
        content_model: ContentBasedRecommender,
        als_weight: float = 0.90,
        missing_content_policy: str = "neutral",
    ) -> None:
        self.als_model = als_model
        self.content_model = content_model
        self._als_weight = 0.0
        self.als_weight = als_weight
        self._missing_content_policy = "neutral"
        self.missing_content_policy = missing_content_policy

        self._item_index: list[str] = []
        self._item_pos: dict[str, int] = {}
        self._user_ids: list[str] = []
        self._user_pos: dict[str, int] = {}
        self._seen_positions: list[np.ndarray] = []
        self._content_available: Optional[np.ndarray] = None

        # Full-catalogue normalised component scores.  float32 keeps the two
        # 5,199 x 8,640 matrices to roughly 360 MB in total.
        self._als_norm: Optional[np.ndarray] = None
        self._content_norm: Optional[np.ndarray] = None

    @property
    def als_weight(self) -> float:
        return self._als_weight

    @als_weight.setter
    def als_weight(self, value: float) -> None:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("als_weight must be in [0, 1].")
        self._als_weight = value

    @property
    def content_weight(self) -> float:
        return 1.0 - self._als_weight

    @property
    def missing_content_policy(self) -> str:
        return self._missing_content_policy

    @missing_content_policy.setter
    def missing_content_policy(self, value: str) -> None:
        if value not in {"neutral", "penalize"}:
            raise ValueError("missing_content_policy must be 'neutral' or 'penalize'.")
        self._missing_content_policy = value

    @property
    def name(self) -> str:
        return (
            f"Hybrid(ALS={self.als_weight:.3f},"
            f"TFIDF={self.content_weight:.3f},minmax,"
            f"missing={self.missing_content_policy})"
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> "HybridRecommender":
        """Fit both component models, align their catalogues, and cache scores."""
        self.als_model.fit(train_df, item_features_df)
        self.content_model.fit(train_df, item_features_df)

        self._item_index = list(self.content_model._item_index)
        self._item_pos = {iid: i for i, iid in enumerate(self._item_index)}
        self._content_available = np.asarray(
            self.content_model._item_vectors.getnnz(axis=1) > 0,
            dtype=bool,
        )
        als_items = set(self.als_model._col_to_item_id)
        content_items = set(self._item_index)
        if als_items != content_items:
            only_als = len(als_items - content_items)
            only_content = len(content_items - als_items)
            raise ValueError(
                "ALS and TF-IDF catalogues differ "
                f"(ALS-only={only_als}, TFIDF-only={only_content})."
            )

        self._user_ids = sorted(train_df["user_id_raw"].unique())
        self._user_pos = {uid: i for i, uid in enumerate(self._user_ids)}
        self._seen_positions = []
        for uid in self._user_ids:
            seen = self.content_model._user_seen.get(uid, set())
            self._seen_positions.append(
                np.fromiter(
                    (self._item_pos[iid] for iid in seen if iid in self._item_pos),
                    dtype=np.int32,
                )
            )

        self._prepare_score_cache()
        return self

    @staticmethod
    def _minmax_unseen(scores: np.ndarray, seen: np.ndarray) -> np.ndarray:
        """Min-max normalise one user row, leaving seen items unavailable."""
        result = np.asarray(scores, dtype=np.float32).copy()
        if seen.size:
            result[seen] = np.nan
        finite = np.isfinite(result)
        if not finite.any():
            return np.full(result.shape, -np.inf, dtype=np.float32)

        low = float(np.min(result[finite]))
        high = float(np.max(result[finite]))
        if high > low:
            result[finite] = (result[finite] - low) / (high - low)
        else:
            result[finite] = 0.0
        result[~finite] = -np.inf
        return result

    def _prepare_score_cache(self) -> None:
        """Compute aligned, normalised ALS and content scores once per user."""
        if self.als_model._scores is None:
            raise RuntimeError("ALS component is not fitted.")
        if self.content_model._user_profiles is None:
            raise RuntimeError("Content component is not fitted.")

        n_users = len(self._user_ids)
        n_items = len(self._item_index)
        self._als_norm = np.empty((n_users, n_items), dtype=np.float32)
        self._content_norm = np.empty((n_users, n_items), dtype=np.float32)

        # ALS columns are lexicographically sorted raw IDs; TF-IDF rows use
        # preprocessing order.  Align explicitly instead of relying on the
        # current dataset happening to contain the same order.
        als_cols = np.asarray(
            [self.als_model._item_id_to_col[iid] for iid in self._item_index],
            dtype=np.int32,
        )

        log.info(
            "Preparing exhaustive hybrid score cache: users=%d items=%d",
            n_users,
            n_items,
        )
        for row, uid in enumerate(self._user_ids):
            als_user_row = self.als_model._user_id_to_row[uid]
            content_user_row = self.content_model._user_id_to_row[uid]
            als_raw = self.als_model._scores[als_user_row, als_cols]
            content_raw = np.asarray(
                (
                    self.content_model._user_profiles[content_user_row]
                    @ self.content_model._item_vectors.T
                ).todense()
            ).ravel()
            seen = self._seen_positions[row]
            self._als_norm[row] = self._minmax_unseen(als_raw, seen)
            self._content_norm[row] = self._minmax_unseen(content_raw, seen)
            if (row + 1) % 500 == 0 or row + 1 == n_users:
                log.info("  cached component scores for %d/%d users", row + 1, n_users)

    def component_scores(
        self,
        user_id_raw: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return read-only normalised ALS and TF-IDF score rows for analysis."""
        if self._als_norm is None or self._content_norm is None:
            raise RuntimeError("Call fit() before requesting component scores.")
        if user_id_raw not in self._user_pos:
            raise KeyError(f"User {user_id_raw!r} was not seen during training.")
        row = self._user_pos[user_id_raw]
        als = self._als_norm[row].view()
        content = self._content_norm[row].view()
        als.flags.writeable = False
        content.flags.writeable = False
        return als, content

    def recommend(
        self,
        user_id_raw: str,
        n: int,
        exclude_seen: bool = True,
    ) -> list[tuple[str, float]]:
        if self._als_norm is None or self._content_norm is None:
            raise RuntimeError("Call fit() before recommend().")
        if user_id_raw not in self._user_pos:
            raise KeyError(f"User {user_id_raw!r} was not seen during training.")
        if n <= 0:
            return []

        row = self._user_pos[user_id_raw]
        als = self._als_norm[row]
        content = self._content_norm[row]
        valid = np.isfinite(als) & np.isfinite(content)
        scores = np.full(als.shape, -np.inf, dtype=np.float32)
        scores[valid] = (
            self.als_weight * als[valid]
            + self.content_weight * content[valid]
        )
        if self.missing_content_policy == "neutral":
            if self._content_available is None:
                raise RuntimeError("Content-availability mask is unavailable.")
            # A zero TF-IDF vector is absent evidence, not evidence of
            # dislike.  Keep the ALS score unchanged for those items.
            missing = valid & ~self._content_available
            scores[missing] = als[missing]

        # The cache always masks training items.  Reconstruct their component
        # blend only for the uncommon interactive exclude_seen=False path.
        if not exclude_seen:
            seen = self._seen_positions[row]
            if seen.size:
                # Offline evaluation always uses exclude_seen=True.  Raising
                # here is safer than silently returning a partially scored set.
                raise NotImplementedError(
                    "Hybrid score cache is built for exclude_seen=True."
                )

        n_available = int(np.isfinite(scores).sum())
        if n_available == 0:
            return []
        n_take = min(int(n), n_available)
        if n_take == scores.size:
            candidates = np.arange(scores.size)
        else:
            candidates = np.argpartition(-scores, kth=n_take - 1)[:n_take]
        order = np.lexsort(
            (candidates, -scores[candidates].astype(np.float64))
        )
        top = candidates[order][:n_take]
        return [(self._item_index[int(i)], float(scores[i])) for i in top]

    def explain(self, user_id_raw: str, item_id_raw: str) -> str:
        if item_id_raw not in self._item_pos:
            return f"Unknown item {item_id_raw!r} -- cannot explain."
        als, content = self.component_scores(user_id_raw)
        pos = self._item_pos[item_id_raw]
        if not np.isfinite(als[pos]) or not np.isfinite(content[pos]):
            return "This item is already in the user's training history."
        blended = self.als_weight * als[pos] + self.content_weight * content[pos]
        if (
            self.missing_content_policy == "neutral"
            and self._content_available is not None
            and not self._content_available[pos]
        ):
            blended = float(als[pos])
            return (
                f"Hybrid score {blended:.3f}: ALS supplies the ranking signal "
                "because this item has no usable tags; missing content evidence "
                "is treated as neutral rather than negative."
            )
        return (
            f"Hybrid score {blended:.3f}: {self.als_weight:.1%} ALS latent "
            f"match ({als[pos]:.3f}) plus {self.content_weight:.1%} tag "
            f"TF-IDF match ({content[pos]:.3f}), after per-user normalisation."
        )
