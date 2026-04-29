"""
Collaborative Filtering Recommender (Implicit Matrix Factorisation).

Algorithm
---------
Hu, Koren & Volinsky (2008) Alternating Least Squares for implicit feedback
("Collaborative Filtering for Implicit Feedback Datasets", ICDM 2008).

Given the binary user-item interaction matrix R (n_users × n_items):

    minimise_{P, Q}   Σ_{u,i} c_ui (p_ui - p_u · q_i)^2 + λ(‖P‖_F^2 + ‖Q‖_F^2)

with
    p_ui = 1 if r_ui > 0 else 0           (binary preference)
    c_ui = 1 + α · r_ui                    (confidence: observed = 1 + α, unseen = 1)

At each ALS half-step, with the other factor matrix held fixed, every row
p_u (resp. q_i) has a closed-form normal-equation solution. For binary
implicit data the HKV 2008 identity lets us avoid materialising the
n_items × n_items confidence matrix per user:

    A_u = Y^T Y  +  α · Y_Ω_u^T Y_Ω_u  +  λ I
    b_u = (1 + α) · Σ_{i ∈ Ω_u} Y_i       where Ω_u = user u's observed items
    p_u = A_u^{-1} b_u

Y^T Y is shared across users and precomputed once per half-step; the per-user
term Y_Ω_u^T Y_Ω_u touches only rows of observed items, so each solve is
O(|Ω_u| · d^2 + d^3). With d = 64 this is microseconds per user in NumPy.

Why not the `implicit` library
------------------------------
The `implicit` package (a C++/Cython ALS implementation) is the reference
production choice, and the project's ``requirements.txt`` originally listed
it. It has no pre-built wheel for Python 3.13 on Windows and building from
source requires MSVC, which is not part of the project toolchain. Rather
than introducing an opaque binary dependency that the rest of the pipeline
can't be reasoned about from, the algorithm is implemented here directly.

Practical consequences:
  * One fewer environment dependency. The repo now trains ALS on any
    machine that can run NumPy + SciPy.
  * Full determinism is under our control (a single NumPy RNG seed governs
    factor initialisation; no library-side threads reorder anything).
  * Training is slower than `implicit`'s parallel Cython inner loop but
    still finishes in well under a minute for 5 199 × 8 640 × d = 64.
  * Numerical behaviour matches HKV 2008 exactly; this is the same
    objective `implicit` optimises.

Hyperparameters (config.py)
---------------------------
    CF_FACTORS          latent dim d        (64)
    CF_ALPHA            confidence scale α  (40)
    CF_REGULARIZATION   ridge λ             (0.01)
    CF_ITERATIONS       ALS sweeps          (20)
    SPLIT_SEED          RNG seed            (42)

Cold-start
----------
Users not seen during training cannot be embedded; ``recommend()`` raises
``KeyError`` and the evaluator records the failure. In this dataset every
retained user appears in training, so this path is not exercised offline.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

import config
from src.recommenders.base import BaseRecommender

log = logging.getLogger(__name__)


class CollaborativeFilteringRecommender(BaseRecommender):
    """Implicit-feedback ALS collaborative filter (pure NumPy/SciPy)."""

    def __init__(
        self,
        factors: int = config.CF_FACTORS,
        regularization: float = config.CF_REGULARIZATION,
        iterations: int = config.CF_ITERATIONS,
        alpha: float = config.CF_ALPHA,
        random_state: int = config.SPLIT_SEED,
    ) -> None:
        self._factors = int(factors)
        self._regularization = float(regularization)
        self._iterations = int(iterations)
        self._alpha = float(alpha)
        self._random_state = int(random_state)

        # Learned state (populated in fit)
        self._user_factors: Optional[np.ndarray] = None   # (n_users, d)
        self._item_factors: Optional[np.ndarray] = None   # (n_items, d)
        self._scores: Optional[np.ndarray] = None         # (n_users, n_items)

        # ID bookkeeping
        self._user_id_to_row: dict[str, int] = {}
        self._item_id_to_col: dict[str, int] = {}
        self._col_to_item_id: list[str] = []
        self._items_array: Optional[np.ndarray] = None

        # Sparse training matrix + per-user observed items (for recommend())
        self._user_items: Optional[sp.csr_matrix] = None
        self._user_seen: dict[str, set[str]] = {}

    @property
    def name(self) -> str:
        return f"CollaborativeFiltering(ALS, d={self._factors})"

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> "CollaborativeFilteringRecommender":
        """Build the user-item matrix and run ``iterations`` of ALS on it.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training interactions. Must contain ``user_id_raw`` and
            ``item_id_raw`` (the pre-remapped ``*_idx`` columns are not
            used here; we build our own local, dense index to make the
            model self-contained).
        item_features_df : pd.DataFrame
            Unused (CF is content-agnostic), kept for interface parity.
        """
        # --- Local index: compact integers derived from raw ids ------------
        # Sorting makes the row/col ordering reproducible across runs and
        # gives a natural, deterministic tie-break (lower item_id_raw first)
        # that matches the popularity baseline's tie-break policy.
        users = sorted(train_df["user_id_raw"].unique())
        items = sorted(train_df["item_id_raw"].unique())
        self._user_id_to_row = {u: i for i, u in enumerate(users)}
        self._item_id_to_col = {it: j for j, it in enumerate(items)}
        self._col_to_item_id = list(items)
        self._items_array = np.asarray(self._col_to_item_id, dtype=object)

        n_users, n_items = len(users), len(items)

        rows = train_df["user_id_raw"].map(self._user_id_to_row).to_numpy()
        cols = train_df["item_id_raw"].map(self._item_id_to_col).to_numpy()
        data = np.ones(len(train_df), dtype=np.float32)

        user_items = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )
        user_items.sum_duplicates()
        user_items.data[:] = 1.0   # enforce binary preference (no double-counting)
        item_users = user_items.T.tocsr()
        self._user_items = user_items

        # Cache per-user observed raw-ids for the explain() path
        self._user_seen = (
            train_df.groupby("user_id_raw")["item_id_raw"]
            .apply(set)
            .to_dict()
        )

        # --- Factor initialisation -----------------------------------------
        # Small-variance Gaussian so the first update is well-conditioned
        # (the normal-equation system has λI on the diagonal regardless).
        rng = np.random.default_rng(self._random_state)
        scale = 0.01
        self._user_factors = rng.normal(
            0.0, scale, size=(n_users, self._factors)
        ).astype(np.float32)
        self._item_factors = rng.normal(
            0.0, scale, size=(n_items, self._factors)
        ).astype(np.float32)

        # --- ALS loop -------------------------------------------------------
        log.info(
            "Fitting ALS: users=%d items=%d  alpha=%g lambda=%g d=%d iters=%d",
            n_users, n_items, self._alpha, self._regularization,
            self._factors, self._iterations,
        )
        for it in range(1, self._iterations + 1):
            t0 = time.perf_counter()
            self._als_half_step(user_items, self._user_factors, self._item_factors)
            t1 = time.perf_counter()
            self._als_half_step(item_users, self._item_factors, self._user_factors)
            t2 = time.perf_counter()
            log.info(
                "  iter %2d/%d  user-update %.2fs  item-update %.2fs",
                it, self._iterations, t1 - t0, t2 - t1,
            )

        # --- Precompute the full score matrix once -------------------------
        # (n_users, n_items) float32 ≈ 180 MB at this size. Done once at fit
        # time so per-user recommend() is a single memory read + mask + top-K.
        # Float32 is deliberate: ALS scores are low-rank and noisy, double
        # precision is not informative here.
        self._scores = self._user_factors @ self._item_factors.T

        log.info(
            "ALS fitted. Score matrix shape=%s  (users=%d × items=%d)",
            self._scores.shape, n_users, n_items,
        )
        return self

    # ------------------------------------------------------------------
    # ALS half-step (HKV 2008)
    # ------------------------------------------------------------------

    def _als_half_step(
        self,
        R_csr: sp.csr_matrix,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        """Update left-factor matrix X in place, holding right-factor Y fixed.

        Closed-form per-row solve (HKV 2008 identity for binary r_ui):

            A_u = Y^T Y + α · Y_Ω^T Y_Ω + λ I
            b_u = (1 + α) · Σ_{i ∈ Ω}  Y_i
            X_u = A_u^{-1} b_u
        """
        alpha = self._alpha
        reg = self._regularization
        d = Y.shape[1]

        # Precompute the dense d×d term shared across all rows of X.
        YtY = (Y.T @ Y).astype(np.float64)
        reg_I = reg * np.eye(d, dtype=np.float64)
        base = YtY + reg_I

        indptr = R_csr.indptr
        indices = R_csr.indices

        Y64 = Y.astype(np.float64)   # do solves in float64 for conditioning
        n_rows = R_csr.shape[0]

        for u in range(n_rows):
            start, end = indptr[u], indptr[u + 1]
            if start == end:
                # Cold row: no observations → regularised solve drives X_u → 0.
                X[u].fill(0.0)
                continue
            obs = indices[start:end]
            Y_obs = Y64[obs]                       # (|Ω_u|, d)
            A = base + alpha * (Y_obs.T @ Y_obs)
            b = (1.0 + alpha) * Y_obs.sum(axis=0)  # (d,)
            X[u] = np.linalg.solve(A, b).astype(X.dtype, copy=False)

    # ------------------------------------------------------------------
    # Recommend
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id_raw: str,
        n: int,
        exclude_seen: bool = True,
    ) -> list[tuple[str, float]]:
        if self._scores is None:
            raise RuntimeError("Call fit() before recommend().")
        if user_id_raw not in self._user_id_to_row:
            raise KeyError(f"User {user_id_raw!r} was not seen during training.")

        u = self._user_id_to_row[user_id_raw]
        scores = self._scores[u].copy()   # (n_items,) float32

        if exclude_seen and self._user_items is not None:
            seen_cols = self._user_items.indices[
                self._user_items.indptr[u] : self._user_items.indptr[u + 1]
            ]
            scores[seen_cols] = -np.inf

        # Deterministic top-N.
        # Primary: score desc.
        # Secondary: item-column index asc (== item_id_raw asc because the
        # column order was built from sorted(items)). This matches the
        # popularity baseline's tie-break policy. Exact ties on ALS scores
        # are floating-point-rare but we lock the policy anyway.
        if n >= scores.size:
            order = np.lexsort((np.arange(scores.size), -scores.astype(np.float64)))
            return [
                (self._col_to_item_id[int(i)], float(scores[i]))
                for i in order
                if np.isfinite(scores[i])
            ][:n]

        # Candidate set: the top-(n + margin) by argpartition, then exact sort.
        # Margin of 0 is fine since np.argpartition is exact at the boundary.
        part = np.argpartition(-scores, kth=n - 1)[:n]
        # Exact sort of the small partition with lexsort tie-break.
        part_order = np.lexsort(
            (part, -scores[part].astype(np.float64))
        )
        top = part[part_order]
        return [
            (self._col_to_item_id[int(i)], float(scores[i]))
            for i in top
            if np.isfinite(scores[i])
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    #
    # The personalization layer only needs the *item* side of the
    # trained model: item factors, item column order, and the
    # hyperparameters used at training time (so fold-in uses the same
    # alpha and lambda). We deliberately do NOT re-save user factors;
    # an interactive user gets their own latent vector via fold-in
    # against the frozen item factors.

    def save_state(self, path) -> None:
        """Persist the minimal state needed by the personalization layer.

        Saves: item_factors, item id ordering, alpha, reg, iters, d.
        User factors and the full score matrix are NOT saved -- both
        can be rebuilt at any time by re-running ``fit``.
        """
        import pickle
        from pathlib import Path

        if self._item_factors is None or self._items_array is None:
            raise RuntimeError("Call fit() before save_state().")

        state = {
            "item_factors": self._item_factors.astype(np.float32, copy=False),
            "item_ids": list(self._col_to_item_id),
            "alpha": self._alpha,
            "reg": self._regularization,
            "iterations": self._iterations,
            "factors": self._factors,
            "random_state": self._random_state,
            "_schema_version": 1,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Saved ALS state -> %s (%d items, d=%d)",
                 path, len(state["item_ids"]), state["factors"])

    @staticmethod
    def load_state(path) -> dict:
        """Load a state dict previously written by :meth:`save_state`."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain(self, user_id_raw: str, item_id_raw: str) -> str:
        if self._user_factors is None or self._item_factors is None:
            return "Model not yet fitted."
        if user_id_raw not in self._user_id_to_row:
            return f"Unknown user {user_id_raw!r} — cannot explain."
        if item_id_raw not in self._item_id_to_col:
            return f"Unknown item {item_id_raw!r} — cannot explain."
        u = self._user_id_to_row[user_id_raw]
        i = self._item_id_to_col[item_id_raw]
        score = float(self._user_factors[u] @ self._item_factors[i])
        return (
            f"ALS predicts a latent-taste match score of {score:.3f} for "
            f"this track (computed in a {self._factors}-dim factor space "
            f"learned from {self._iterations} alternating least-squares sweeps "
            f"over users' listening histories)."
        )
