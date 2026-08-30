"""EASE: Embarrassingly Shallow Autoencoder for implicit recommendation."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.recommenders.base import BaseRecommender

log = logging.getLogger(__name__)


class EASERecommender(BaseRecommender):
    """Closed-form item-item linear recommender from Steck (2019)."""

    def __init__(self, regularization: float = 300.0) -> None:
        self.regularization = float(regularization)
        self._user_id_to_row: dict[str, int] = {}
        self._item_id_to_col: dict[str, int] = {}
        self._col_to_item_id: list[str] = []
        self._user_items: Optional[sp.csr_matrix] = None
        self._weights: Optional[np.ndarray] = None
        self._scores: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return f"EASE(lambda={self.regularization:g})"

    def fit(
        self,
        train_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> "EASERecommender":
        users = sorted(train_df["user_id_raw"].unique())
        items = sorted(train_df["item_id_raw"].unique())
        self._user_id_to_row = {uid: i for i, uid in enumerate(users)}
        self._item_id_to_col = {iid: i for i, iid in enumerate(items)}
        self._col_to_item_id = list(items)

        rows = train_df["user_id_raw"].map(self._user_id_to_row).to_numpy()
        cols = train_df["item_id_raw"].map(self._item_id_to_col).to_numpy()
        X = sp.csr_matrix(
            (np.ones(len(train_df), dtype=np.float32), (rows, cols)),
            shape=(len(users), len(items)),
            dtype=np.float32,
        )
        X.sum_duplicates()
        X.data[:] = 1.0
        self._user_items = X

        log.info(
            "Fitting EASE: users=%d items=%d lambda=%g",
            X.shape[0], X.shape[1], self.regularization,
        )
        # EASE closed form:
        #   P = (X^T X + lambda I)^-1
        #   B_ij = -P_ij / P_jj, B_jj = 0
        gram = (X.T @ X).toarray().astype(np.float64, copy=False)
        diagonal = np.diag_indices_from(gram)
        gram[diagonal] += self.regularization
        precision = np.linalg.inv(gram)
        diag_precision = np.diag(precision).copy()
        weights = precision / (-diag_precision[np.newaxis, :])
        weights[diagonal] = 0.0
        self._weights = weights.astype(np.float32)
        del gram, precision, weights

        self._scores = np.asarray(X @ self._weights, dtype=np.float32)
        log.info("EASE fitted; score matrix=%s", self._scores.shape)
        return self

    def recommend(
        self,
        user_id_raw: str,
        n: int,
        exclude_seen: bool = True,
    ) -> list[tuple[str, float]]:
        if self._scores is None or self._user_items is None:
            raise RuntimeError("Call fit() before recommend().")
        if user_id_raw not in self._user_id_to_row:
            raise KeyError(f"Unknown user {user_id_raw!r}.")
        if n <= 0:
            return []
        row = self._user_id_to_row[user_id_raw]
        scores = self._scores[row].copy()
        if exclude_seen:
            seen = self._user_items.indices[
                self._user_items.indptr[row] : self._user_items.indptr[row + 1]
            ]
            scores[seen] = -np.inf
        n_take = min(n, int(np.isfinite(scores).sum()))
        if n_take == 0:
            return []
        candidates = np.argpartition(-scores, kth=n_take - 1)[:n_take]
        order = np.lexsort((candidates, -scores[candidates].astype(np.float64)))
        top = candidates[order]
        return [
            (self._col_to_item_id[int(pos)], float(scores[pos])) for pos in top
        ]

    def explain(self, user_id_raw: str, item_id_raw: str) -> str:
        if self._weights is None or self._user_items is None:
            return "Model not yet fitted."
        if user_id_raw not in self._user_id_to_row or item_id_raw not in self._item_id_to_col:
            return "Unknown user or item."
        user_row = self._user_id_to_row[user_id_raw]
        item_col = self._item_id_to_col[item_id_raw]
        seen = self._user_items.indices[
            self._user_items.indptr[user_row] : self._user_items.indptr[user_row + 1]
        ]
        contributions = self._weights[seen, item_col]
        if contributions.size == 0:
            return "No training-history items contribute to this recommendation."
        top_local = np.argsort(-contributions)[:3]
        evidence = [self._col_to_item_id[int(seen[pos])] for pos in top_local]
        return (
            "EASE recommends this item from learned item-to-item co-listening "
            f"weights; strongest history evidence comes from items {evidence}."
        )
