"""
Content-Based Recommender.

Design
------
Scores every catalogue item by cosine similarity to a user's "taste
profile" -- a vector built from the tag-TF-IDF representations of the
items the user interacted with during training. The top-K items (after
excluding the user's training history) are returned as recommendations.

Feature space (tags only, per the project decision)
---------------------------------------------------
The item feature matrix used here is the **preprocessed tag TF-IDF
matrix** produced by the preprocessing stage
(``config.TFIDF_MATRIX_FILE``), not a vectoriser rebuilt at fit time.

That has three consequences that matter for the experiment:

1.  The TF-IDF vocabulary, IDF weights, and sublinear-TF flag were
    learned on training items only. Using the same matrix here removes
    any risk of a second, inconsistent vectoriser being fit on data the
    diversity metric doesn't see.
2.  The Intra-List Diversity metric is measured in this exact same
    space. CB's diversity score is therefore measured in the space it
    actually scores candidates in -- which is the only honest way to
    report intra-list diversity for a content model.
3.  There is a single source of truth for "the tag feature space".
    Popularity and ALS use it only for the diversity metric; CB uses it
    for scoring too.

Profile aggregation
-------------------
For each user u, profile_u = L2-normalise(mean_{i in train(u)} item_i).
Uniform weighting; no timestamp-based decay because the dataset has no
timestamps. Items with zero tag vectors (744 of the 8640) contribute
zero mass to the numerator; after re-normalisation they do not bias the
profile direction, only dilute its magnitude before re-norm.

Scoring
-------
Because user profiles and item vectors are both L2-normalised,
cosine(profile_u, item_i) = profile_u . item_i. We score every item
in the catalogue (8,640 items) with one sparse matrix-vector product
per user. No approximate nearest-neighbour index is needed at this
scale.

Why content-based here is methodologically distinct from CF
-----------------------------------------------------------
Tag files were scraped from Last.fm metadata, not derived from this
dataset's interaction matrix, so ranking items by tag similarity does
not circularly re-encode the same signal CF is using. This is the
point of the three-way comparison: popularity uses the marginal item
distribution, CF uses co-listening structure, CB uses external content
metadata. Any divergence in their beyond-accuracy profiles reflects the
information source, not the model family.

Cold-start behaviour
--------------------
- Item cold-start: catalogue items with zero-vector tag representation
  (744 items) score 0 against every non-orthogonal user profile; they
  almost never enter the top-K. This is reported, not imputed.
- User cold-start: not exercised offline (all retained users appear in
  training), but supported in principle: a profile can be seeded from
  any item subset.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

import config
from src.recommenders.base import BaseRecommender

log = logging.getLogger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """Tag-TF-IDF content-based recommender.

    Parameters
    ----------
    item_vectors : sp.csr_matrix
        (n_items, n_features) item feature matrix. Must be L2-normalised
        row-wise so cosine similarity reduces to a dot product. This is
        the tag TF-IDF matrix produced by the preprocessor.
    item_index : list[str]
        Row order of ``item_vectors`` expressed as ``item_id_raw``
        strings. Must have length == item_vectors.shape[0].
    feature_mode : str
        Documented as ``"tags_bow"`` for this project; retained in the
        class signature for forward compatibility but only the precomputed
        vector path is supported now.
    """

    def __init__(
        self,
        item_vectors: sp.csr_matrix,
        item_index: list[str],
        feature_mode: str = config.CB_FEATURE_MODE,
    ) -> None:
        if item_vectors.shape[0] != len(item_index):
            raise ValueError(
                f"item_vectors has {item_vectors.shape[0]} rows but "
                f"item_index has {len(item_index)} entries."
            )

        # Guard: ensure L2-normalised. We re-normalise defensively so the
        # caller cannot break the cosine invariant even by accident.
        self._item_vectors: sp.csr_matrix = normalize(item_vectors, norm="l2").tocsr()
        self._item_index: list[str] = list(item_index)
        self._item_pos: dict[str, int] = {
            iid: i for i, iid in enumerate(self._item_index)
        }
        self._feature_mode = feature_mode

        # Populated in fit()
        self._user_id_to_row: dict[str, int] = {}
        self._user_profiles: Optional[sp.csr_matrix] = None     # (n_users, n_features)
        self._user_items: Optional[sp.csr_matrix] = None        # (n_users, n_items)
        self._user_seen: dict[str, set[str]] = {}

    @property
    def name(self) -> str:
        return f"ContentBased(mode={self._feature_mode})"

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> "ContentBasedRecommender":
        """Build per-user profile vectors in the tag-TF-IDF space.

        ``item_features_df`` is accepted for interface parity but is not
        used: the feature space is fully defined by ``item_vectors`` and
        ``item_index`` that were passed to ``__init__``.
        """
        log.info(
            "Fitting ContentBasedRecommender (mode=%s)  items=%d features=%d",
            self._feature_mode,
            self._item_vectors.shape[0],
            self._item_vectors.shape[1],
        )

        users = sorted(train_df["user_id_raw"].unique())
        self._user_id_to_row = {u: i for i, u in enumerate(users)}
        n_users = len(users)
        n_items = self._item_vectors.shape[0]

        # Build a binary user-item interaction matrix restricted to items
        # that have a row in the feature space. In this dataset every
        # training item is in the feature space, so no rows are dropped,
        # but the explicit .map(...) with NaN filtering keeps this honest
        # if the two artefacts ever go out of sync.
        mapped_items = train_df["item_id_raw"].map(self._item_pos)
        mask = mapped_items.notna()
        if not mask.all():
            dropped = (~mask).sum()
            log.warning(
                "Dropping %d training interactions whose items are absent "
                "from the content feature space.", dropped,
            )

        rows = train_df.loc[mask, "user_id_raw"].map(self._user_id_to_row).to_numpy()
        cols = mapped_items.loc[mask].astype(np.int64).to_numpy()
        data = np.ones(len(rows), dtype=np.float32)

        user_items = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )
        user_items.sum_duplicates()
        user_items.data[:] = 1.0   # enforce binary preference
        self._user_items = user_items

        # Mean-aggregate the tag vectors of each user's observed items,
        # then L2-normalise. Doing it as two sparse matmuls is a single
        # scipy operation per step and avoids a per-user Python loop:
        #
        #   user_items_l1 = rowwise L1-normalised user_items
        #                 = user_items[u, i] = 1 / |train(u)|
        #   user_profiles = user_items_l1 @ item_vectors
        #                 = mean of item vectors for each user's train set
        #   user_profiles = rowwise L2-normalised user_profiles
        user_items_l1 = normalize(user_items, norm="l1", axis=1)
        user_profiles = user_items_l1 @ self._item_vectors    # sparse CSR
        user_profiles = normalize(user_profiles, norm="l2", axis=1).tocsr()
        self._user_profiles = user_profiles

        # Cache per-user observed items for the exclude-seen filter at
        # recommend time.
        self._user_seen = (
            train_df.groupby("user_id_raw")["item_id_raw"]
            .apply(set)
            .to_dict()
        )

        log.info(
            "User profiles built: shape=%s  nnz=%d  zero-profile users=%d",
            user_profiles.shape,
            user_profiles.nnz,
            int((user_profiles.getnnz(axis=1) == 0).sum()),
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
        if self._user_profiles is None:
            raise RuntimeError("Call fit() before recommend().")
        if user_id_raw not in self._user_id_to_row:
            raise KeyError(f"User {user_id_raw!r} was not seen during training.")

        u = self._user_id_to_row[user_id_raw]
        profile_row = self._user_profiles[u]

        # scores[i] = cosine(profile_u, item_i). Both sides are L2-normalised,
        # so the inner product equals cosine similarity directly.
        # Result is a (1, n_items) sparse row; densify for top-K selection.
        scores = np.asarray(
            (profile_row @ self._item_vectors.T).todense()
        ).ravel().astype(np.float32, copy=False)

        if exclude_seen and self._user_items is not None:
            seen_cols = self._user_items.indices[
                self._user_items.indptr[u] : self._user_items.indptr[u + 1]
            ]
            scores[seen_cols] = -np.inf

        # Deterministic top-N with the same tie-break policy as the other
        # two models: primary = score descending, secondary = item column
        # index ascending (equivalent to item_id_raw ascending because the
        # feature-matrix ordering is the item_id order saved by
        # preprocessing).
        if n >= scores.size:
            order = np.lexsort(
                (np.arange(scores.size), -scores.astype(np.float64))
            )
            return [
                (self._item_index[int(i)], float(scores[i]))
                for i in order
                if np.isfinite(scores[i])
            ][:n]

        part = np.argpartition(-scores, kth=n - 1)[:n]
        part_order = np.lexsort((part, -scores[part].astype(np.float64)))
        top = part[part_order]
        return [
            (self._item_index[int(i)], float(scores[i]))
            for i in top
            if np.isfinite(scores[i])
        ]

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain(self, user_id_raw: str, item_id_raw: str) -> str:
        if self._user_profiles is None:
            return "Model not yet fitted."
        if user_id_raw not in self._user_id_to_row:
            return f"Unknown user {user_id_raw!r} -- cannot explain."
        if item_id_raw not in self._item_pos:
            return f"Unknown item {item_id_raw!r} -- not in the content index."

        u = self._user_id_to_row[user_id_raw]
        i = self._item_pos[item_id_raw]
        sim = float(
            (self._user_profiles[u] @ self._item_vectors[i].T).toarray().ravel()[0]
        )
        n_seen = len(self._user_seen.get(user_id_raw, set()))
        return (
            f"This track's tag profile has cosine similarity {sim:.3f} to "
            f"the average tag profile of the {n_seen} tracks in your "
            f"training history, in a tag-TF-IDF feature space of "
            f"{self._item_vectors.shape[1]} dimensions."
        )
