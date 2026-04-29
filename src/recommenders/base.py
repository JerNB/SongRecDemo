"""
Abstract base class that every recommender in this project must implement.

Contract
--------
All three recommenders (Popularity, CollaborativeFiltering, ContentBased)
expose the same public interface so the evaluation harness can treat them
identically.  This is critical for fair comparison: the evaluator never
calls model-specific methods; it only calls ``fit`` and ``recommend``.

Interface summary
-----------------
- ``fit(train_df, item_features_df)``
    Receives the training interaction DataFrame and item features; trains
    or indexes whatever internal state the model needs.
- ``recommend(user_id_raw, n, exclude_seen)``
    Returns a ranked list of (item_id_raw, score) tuples of length <= n.
    ``exclude_seen=True`` (default) filters out items the user already
    interacted with in training.
- ``name`` property
    Short identifier used in evaluation tables and plots.
- ``explain(user_id_raw, item_id_raw)``
    Returns a human-readable string explaining why the item was recommended.
    This supports the "why this track?" requirement for the demo interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseRecommender(ABC):
    """Abstract recommender.  Subclass and implement all abstract methods."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short display name, e.g. 'PopularityBaseline'."""

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> "BaseRecommender":
        """Train (or index) the model.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training interactions.  Must have columns:
            ``user_id_raw``, ``item_id_raw``, ``user_idx``, ``item_idx``.
        item_features_df : pd.DataFrame
            Output of ``build_item_features()``.  Index is ``item_id_raw``.
            Columns: ``tags_normalised``, ``desc_clean``, ``has_tags``,
            ``desc_len_tokens``, ``is_desc_duplicate``.

        Returns
        -------
        self : BaseRecommender
            Returns itself to support chaining: ``model.fit(...).recommend(...)``
        """

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @abstractmethod
    def recommend(
        self,
        user_id_raw: str,
        n: int,
        exclude_seen: bool = True,
    ) -> list[tuple[str, float]]:
        """Generate top-n recommendations for one user.

        Parameters
        ----------
        user_id_raw : str
            Raw user identifier (string, not remapped index).
        n : int
            Number of recommendations to return.
        exclude_seen : bool
            If True, items already in the user's training history are
            removed before ranking.  Always True during offline evaluation
            to avoid trivially recovering known interactions.

        Returns
        -------
        list of (item_id_raw: str, score: float)
            Descending score order.  Length <= n (may be shorter if the
            catalogue minus seen items has fewer than n entries).
        """

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    @abstractmethod
    def explain(self, user_id_raw: str, item_id_raw: str) -> str:
        """Return a plain-English explanation for why item was recommended.

        Every model must implement this for the demo interface.  The string
        should be one or two sentences that a non-technical audience can
        understand.

        Examples (expected style, not enforced by the base class):
        - Popularity: "This track was listened to by 423 users in the
          training data, making it one of the most popular items."
        - CF: "Users with similar listening histories to you (e.g. users
          A and B) also liked this track."
        - Content: "This track shares tags ['indie', 'mellow', 'psychedelic']
          with the seed items in your history."
        """

    # ------------------------------------------------------------------
    # Optional helpers (not abstract – subclasses may override)
    # ------------------------------------------------------------------

    def batch_recommend(
        self,
        user_ids: list[str],
        n: int,
        exclude_seen: bool = True,
    ) -> dict[str, list[tuple[str, float]]]:
        """Generate recommendations for multiple users.

        Default implementation loops over ``recommend``; subclasses may
        override with a vectorised version for speed.

        Returns
        -------
        dict[user_id_raw, list of (item_id_raw, score)]
        """
        return {uid: self.recommend(uid, n, exclude_seen) for uid in user_ids}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
