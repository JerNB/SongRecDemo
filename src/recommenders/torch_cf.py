"""PyTorch BPR-MF and LightGCN recommenders for implicit KGRec feedback."""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from src.recommenders.base import BaseRecommender

log = logging.getLogger(__name__)


class _TorchCFBase(BaseRecommender):
    def __init__(
        self,
        factors: int = 64,
        epochs: int = 30,
        learning_rate: float = 0.01,
        regularization: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.factors = int(factors)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.random_state = int(random_state)
        self._user_id_to_row: dict[str, int] = {}
        self._item_id_to_col: dict[str, int] = {}
        self._col_to_item_id: list[str] = []
        self._user_items: Optional[sp.csr_matrix] = None
        self._scores: Optional[np.ndarray] = None

    def _prepare(self, train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        users = sorted(train_df["user_id_raw"].unique())
        items = sorted(train_df["item_id_raw"].unique())
        self._user_id_to_row = {uid: i for i, uid in enumerate(users)}
        self._item_id_to_col = {iid: i for i, iid in enumerate(items)}
        self._col_to_item_id = list(items)
        rows = train_df["user_id_raw"].map(self._user_id_to_row).to_numpy(np.int64).copy()
        cols = train_df["item_id_raw"].map(self._item_id_to_col).to_numpy(np.int64).copy()
        X = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(users), len(items)),
        )
        X.sum_duplicates()
        X.data[:] = 1.0
        self._user_items = X
        return rows, cols

    def _sample_negatives(
        self,
        user_rows: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self._user_items is None:
            raise RuntimeError("Interaction matrix unavailable.")
        n_items = self._user_items.shape[1]
        negatives = rng.integers(0, n_items, size=len(user_rows), dtype=np.int64)
        # Vectorised sparse lookup: non-zero means the sampled item was seen.
        collisions = np.asarray(self._user_items[user_rows, negatives]).ravel() > 0
        while collisions.any():
            negatives[collisions] = rng.integers(
                0, n_items, size=int(collisions.sum()), dtype=np.int64
            )
            collisions = np.asarray(
                self._user_items[user_rows, negatives]
            ).ravel() > 0
        return negatives

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
        row = self._user_id_to_row[user_id_raw]
        scores = self._scores[row].copy()
        if exclude_seen:
            seen = self._user_items.indices[
                self._user_items.indptr[row] : self._user_items.indptr[row + 1]
            ]
            scores[seen] = -np.inf
        n_take = min(max(int(n), 0), int(np.isfinite(scores).sum()))
        if n_take == 0:
            return []
        candidates = np.argpartition(-scores, kth=n_take - 1)[:n_take]
        order = np.lexsort((candidates, -scores[candidates].astype(np.float64)))
        return [
            (self._col_to_item_id[int(pos)], float(scores[pos]))
            for pos in candidates[order]
        ]

    def explain(self, user_id_raw: str, item_id_raw: str) -> str:
        if self._scores is None:
            return "Model not yet fitted."
        if user_id_raw not in self._user_id_to_row or item_id_raw not in self._item_id_to_col:
            return "Unknown user or item."
        score = self._scores[
            self._user_id_to_row[user_id_raw], self._item_id_to_col[item_id_raw]
        ]
        return f"The learned implicit-feedback embedding assigns score {score:.3f}."


class BPRMFRecommender(_TorchCFBase):
    """Matrix factorisation trained with Bayesian Personalized Ranking."""

    def __init__(self, batch_size: int = 8192, **kwargs) -> None:
        super().__init__(**kwargs)
        self.batch_size = int(batch_size)

    @property
    def name(self) -> str:
        return f"BPR-MF(d={self.factors},epochs={self.epochs})"

    def fit(self, train_df: pd.DataFrame, item_features_df: pd.DataFrame):
        user_rows, positive_items = self._prepare(train_df)
        n_users, n_items = self._user_items.shape
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        user_embedding = torch.nn.Embedding(n_users, self.factors)
        item_embedding = torch.nn.Embedding(n_items, self.factors)
        torch.nn.init.normal_(user_embedding.weight, std=0.01)
        torch.nn.init.normal_(item_embedding.weight, std=0.01)
        optimizer = torch.optim.Adam(
            list(user_embedding.parameters()) + list(item_embedding.parameters()),
            lr=self.learning_rate,
        )
        users_t = torch.from_numpy(user_rows)
        positives_t = torch.from_numpy(positive_items)

        log.info(
            "Training %s users=%d items=%d interactions=%d",
            self.name, n_users, n_items, len(user_rows),
        )
        for epoch in range(1, self.epochs + 1):
            t0 = time.perf_counter()
            negatives_t = torch.from_numpy(self._sample_negatives(user_rows, rng))
            permutation = torch.randperm(len(user_rows))
            total_loss = 0.0
            for start in range(0, len(user_rows), self.batch_size):
                idx = permutation[start : start + self.batch_size]
                u = user_embedding(users_t[idx])
                p = item_embedding(positives_t[idx])
                n = item_embedding(negatives_t[idx])
                pos_score = torch.sum(u * p, dim=1)
                neg_score = torch.sum(u * n, dim=1)
                ranking_loss = F.softplus(neg_score - pos_score).mean()
                reg = self.regularization * (
                    u.square().sum() + p.square().sum() + n.square().sum()
                ) / len(idx)
                loss = ranking_loss + reg
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(idx)
            if epoch == 1 or epoch % 5 == 0 or epoch == self.epochs:
                log.info(
                    "  epoch %d/%d loss=%.6f time=%.2fs",
                    epoch, self.epochs, total_loss / len(user_rows),
                    time.perf_counter() - t0,
                )

        with torch.no_grad():
            self._scores = (
                user_embedding.weight @ item_embedding.weight.T
            ).numpy().astype(np.float32, copy=False)
        return self


class LightGCNRecommender(_TorchCFBase):
    """LightGCN with full-batch BPR optimisation on the user-item graph."""

    def __init__(self, layers: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.layers = int(layers)

    @property
    def name(self) -> str:
        return (
            f"LightGCN(d={self.factors},layers={self.layers},epochs={self.epochs})"
        )

    def _normalised_adjacency(self) -> torch.Tensor:
        if self._user_items is None:
            raise RuntimeError("Interaction matrix unavailable.")
        X = self._user_items.tocoo()
        n_users, n_items = X.shape
        user_degree = np.asarray(self._user_items.sum(axis=1)).ravel()
        item_degree = np.asarray(self._user_items.sum(axis=0)).ravel()
        values = 1.0 / np.sqrt(user_degree[X.row] * item_degree[X.col])
        item_nodes = n_users + X.col
        indices = np.vstack([
            np.concatenate([X.row, item_nodes]),
            np.concatenate([item_nodes, X.row]),
        ])
        vals = np.concatenate([values, values]).astype(np.float32)
        return torch.sparse_coo_tensor(
            torch.from_numpy(indices).long(),
            torch.from_numpy(vals),
            size=(n_users + n_items, n_users + n_items),
            check_invariants=False,
        ).coalesce()

    def _propagate(
        self,
        adjacency: torch.Tensor,
        user_embedding: torch.nn.Embedding,
        item_embedding: torch.nn.Embedding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = torch.cat([user_embedding.weight, item_embedding.weight], dim=0)
        layers = [embedding]
        for _ in range(self.layers):
            embedding = torch.sparse.mm(adjacency, embedding)
            layers.append(embedding)
        final = torch.stack(layers, dim=0).mean(dim=0)
        n_users = user_embedding.num_embeddings
        return final[:n_users], final[n_users:]

    def fit(self, train_df: pd.DataFrame, item_features_df: pd.DataFrame):
        user_rows, positive_items = self._prepare(train_df)
        n_users, n_items = self._user_items.shape
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        users_t = torch.from_numpy(user_rows)
        positives_t = torch.from_numpy(positive_items)
        adjacency = self._normalised_adjacency()
        user_embedding = torch.nn.Embedding(n_users, self.factors)
        item_embedding = torch.nn.Embedding(n_items, self.factors)
        torch.nn.init.normal_(user_embedding.weight, std=0.01)
        torch.nn.init.normal_(item_embedding.weight, std=0.01)
        optimizer = torch.optim.Adam(
            list(user_embedding.parameters()) + list(item_embedding.parameters()),
            lr=self.learning_rate,
        )

        log.info(
            "Training %s users=%d items=%d edges=%d",
            self.name, n_users, n_items, len(user_rows),
        )
        for epoch in range(1, self.epochs + 1):
            t0 = time.perf_counter()
            negatives_t = torch.from_numpy(self._sample_negatives(user_rows, rng))
            propagated_users, propagated_items = self._propagate(
                adjacency, user_embedding, item_embedding
            )
            u = propagated_users[users_t]
            p = propagated_items[positives_t]
            n = propagated_items[negatives_t]
            pos_score = torch.sum(u * p, dim=1)
            neg_score = torch.sum(u * n, dim=1)
            ranking_loss = F.softplus(neg_score - pos_score).mean()
            raw_u = user_embedding(users_t)
            raw_p = item_embedding(positives_t)
            raw_n = item_embedding(negatives_t)
            reg = self.regularization * (
                raw_u.square().sum() + raw_p.square().sum() + raw_n.square().sum()
            ) / len(user_rows)
            loss = ranking_loss + reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch == 1 or epoch % 5 == 0 or epoch == self.epochs:
                log.info(
                    "  epoch %d/%d loss=%.6f time=%.2fs",
                    epoch, self.epochs, float(loss.detach()), time.perf_counter() - t0,
                )

        with torch.no_grad():
            final_users, final_items = self._propagate(
                adjacency, user_embedding, item_embedding
            )
            self._scores = (
                final_users @ final_items.T
            ).numpy().astype(np.float32, copy=False)
        return self
